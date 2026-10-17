import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from easy_tpp.model.torch_model.torch_baselayer import ScaledSoftplus
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class _SinTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        half = d_model // 2
        self.register_buffer(
            'freqs',
            torch.exp(torch.arange(0, half) * -(math.log(10000.0) / max(half - 1, 1))),
        )
        self.proj = nn.Linear(1 + 2 * half, d_model)

    def forward(self, t):
        angle = t.unsqueeze(-1) * self.freqs
        pe = torch.cat([t.unsqueeze(-1), torch.sin(angle), torch.cos(angle)], dim=-1)
        return self.proj(pe)


def _masked_mean(x, mask, dim):
    mask_f = mask.float().unsqueeze(-1)
    return (x * mask_f).sum(dim) / (mask_f.sum(dim) + 1e-8)


def _masked_max(x, mask, dim):
    out = x.masked_fill(~mask.bool().unsqueeze(-1), float('-inf')).max(dim).values
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class _EventEncoder(nn.Module):
    def __init__(self, num_channels, d_model, dropout):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.val_proj = nn.Linear(1, d_model)
        self.var_emb = nn.Embedding(num_channels, d_model)
        self.time_emb = _SinTimeEmbedding(d_model)
        self.dt_proj = nn.Linear(1, d_model)
        self.mask_emb = nn.Embedding(2, d_model)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, t, mask):
        bsz, seq_len, channels = x.shape
        valid = mask > 0
        t_min = t.masked_fill(~valid, float('inf')).amin(dim=(1, 2), keepdim=True)
        t_max = t.masked_fill(~valid, float('-inf')).amax(dim=(1, 2), keepdim=True)
        has_valid = valid.sum(dim=(1, 2), keepdim=True) > 0
        t_min = torch.where(has_valid, t_min, torch.zeros_like(t_min))
        t_max = torch.where(has_valid, t_max, torch.ones_like(t_max))
        denom = torch.where((t_max - t_min).abs() < 1e-8, torch.ones_like(t_max), t_max - t_min)
        t_norm = (t - t_min) / denom

        seq_idx = torch.arange(1, seq_len + 1, device=t.device).view(1, seq_len, 1).expand(bsz, seq_len, channels)
        masked_idx = seq_idx.masked_fill(~valid, 0)
        prev_idx = torch.cat([torch.zeros_like(masked_idx[:, :1]), masked_idx[:, :-1]], dim=1).cummax(dim=1).values
        prev_t = t.gather(1, (prev_idx - 1).clamp(min=0))
        dt = torch.where(valid & (prev_idx > 0), t - prev_t, torch.zeros_like(t))
        dt_norm = dt / (denom + 1e-8)

        var_ids = torch.arange(channels, device=x.device)
        emb = (
            self.val_proj(x.unsqueeze(-1))
            + self.var_emb(var_ids).view(1, 1, channels, self.d_model)
            + self.time_emb(t_norm)
            + self.dt_proj(dt_norm.unsqueeze(-1))
            + self.mask_emb(mask.long())
        )
        return self.norm(self.fusion_mlp(emb)), t_norm


class _SparseTokenizer(nn.Module):
    TYPE_EVENT = 0
    TYPE_GAP = 1
    TYPE_VAR = 2

    def __init__(self, num_channels, d_model, max_event_tokens, max_gap_tokens, dropout):
        super().__init__()
        self.num_channels = num_channels
        self.max_event_tokens = max_event_tokens
        self.max_gap_tokens = max_gap_tokens
        self.gap_proj = nn.Linear(6, d_model)
        self.var_summary_proj = nn.Linear(15, d_model)
        self.type_emb = nn.Embedding(3, d_model)
        self.var_emb = nn.Embedding(num_channels, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _pad_token_block(self, x, target_len):
        if x.size(1) >= target_len:
            return x
        return F.pad(x, (0, 0, 0, target_len - x.size(1)))

    def _pad_vector_block(self, x, target_len):
        if x.size(1) >= target_len:
            return x
        return F.pad(x, (0, target_len - x.size(1)))

    def _event_tokens(self, event_emb, t_norm, mask):
        bsz, seq_len, channels, dim = event_emb.shape
        flat_len = seq_len * channels
        k = min(self.max_event_tokens, flat_len)
        event_flat = event_emb.reshape(bsz, flat_len, dim)
        time_flat = t_norm.reshape(bsz, flat_len)
        mask_flat = mask.reshape(bsz, flat_len)
        var_flat = torch.arange(channels, device=event_emb.device).view(1, channels).expand(seq_len, channels)
        var_flat = var_flat.reshape(1, flat_len).expand(bsz, flat_len)
        score = time_flat + mask_flat * 1e6 - (1.0 - mask_flat) * 1e9
        _, idx = torch.topk(score, k, dim=1)
        tokens = event_flat.gather(1, idx.unsqueeze(-1).expand(bsz, k, dim))
        token_time = time_flat.gather(1, idx)
        token_var = var_flat.gather(1, idx)
        token_mask = mask_flat.gather(1, idx)
        return (
            self._pad_token_block(tokens, self.max_event_tokens),
            self._pad_vector_block(token_mask, self.max_event_tokens),
            self._pad_vector_block(token_time, self.max_event_tokens),
            self._pad_vector_block(token_var.float(), self.max_event_tokens).long(),
        )

    def _gap_tokens(self, x, t_norm, mask):
        bsz, seq_len, channels = x.shape
        flat_len = seq_len * channels
        k = min(self.max_gap_tokens, flat_len)
        gap_len = torch.zeros_like(t_norm)
        t_left = torch.zeros_like(t_norm)
        v_left = torch.zeros_like(x)
        pair_mask = torch.zeros_like(mask)
        if seq_len > 1:
            gap_len[:, 1:] = (t_norm[:, 1:] - t_norm[:, :-1]).clamp(min=0)
            t_left[:, 1:] = t_norm[:, :-1]
            v_left[:, 1:] = x[:, :-1]
            pair_mask[:, 1:] = mask[:, 1:] * mask[:, :-1]
        gap_score = (gap_len * pair_mask).reshape(bsz, flat_len)
        _, idx = torch.topk(gap_score, k, dim=1)
        t_left_flat = t_left.reshape(bsz, flat_len)
        t_right_flat = t_norm.reshape(bsz, flat_len)
        v_left_flat = v_left.reshape(bsz, flat_len)
        v_right_flat = x.reshape(bsz, flat_len)
        pair_flat = pair_mask.reshape(bsz, flat_len)
        var_flat = torch.arange(channels, device=x.device).view(1, channels).expand(seq_len, channels)
        var_flat = var_flat.reshape(1, flat_len).expand(bsz, flat_len)

        g_len = gap_score.gather(1, idx)
        g_tl = t_left_flat.gather(1, idx)
        g_tr = t_right_flat.gather(1, idx)
        g_vl = v_left_flat.gather(1, idx)
        g_vr = v_right_flat.gather(1, idx)
        g_var = var_flat.gather(1, idx)
        g_mask = pair_flat.gather(1, idx)
        feat = torch.stack([g_len, g_tl, g_tr, g_vl, g_vr, g_vr - g_vl], dim=-1)
        tokens = self.gap_proj(feat) + self.var_emb(g_var)
        return (
            self._pad_token_block(tokens, self.max_gap_tokens),
            self._pad_vector_block(g_mask, self.max_gap_tokens),
            self._pad_vector_block((g_tl + g_tr) * 0.5, self.max_gap_tokens),
            self._pad_vector_block(g_var.float(), self.max_gap_tokens).long(),
        )

    def _var_tokens(self, x, t_norm, mask):
        bsz, seq_len, channels = x.shape
        obs_count = mask.sum(dim=1)
        has_obs = obs_count > 0
        density = obs_count / (seq_len + 1e-8)
        mean_v = (x * mask).sum(dim=1) / (obs_count + 1e-8)
        centered = (x - mean_v.unsqueeze(1)) * mask
        std_v = torch.sqrt((centered * centered).sum(dim=1) / (obs_count + 1e-8) + 1e-8)
        min_v = x.masked_fill(mask == 0, float('inf')).amin(dim=1)
        max_v = x.masked_fill(mask == 0, float('-inf')).amax(dim=1)
        first_idx = t_norm.masked_fill(mask == 0, float('inf')).argmin(dim=1)
        last_idx = t_norm.masked_fill(mask == 0, float('-inf')).argmax(dim=1)
        first_v = x.gather(1, first_idx.unsqueeze(1)).squeeze(1)
        last_v = x.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        first_t = t_norm.gather(1, first_idx.unsqueeze(1)).squeeze(1)
        last_t = t_norm.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        zeros = torch.zeros_like(mean_v)
        first_v = torch.where(has_obs, first_v, zeros)
        last_v = torch.where(has_obs, last_v, zeros)
        first_t = torch.where(has_obs, first_t, zeros)
        last_t = torch.where(has_obs, last_t, zeros)
        min_v = torch.where(has_obs, min_v, zeros)
        max_v = torch.where(has_obs, max_v, zeros)
        gaps = torch.zeros_like(t_norm)
        pair_mask = torch.zeros_like(mask)
        if seq_len > 1:
            gaps[:, 1:] = (t_norm[:, 1:] - t_norm[:, :-1]).clamp(min=0)
            pair_mask[:, 1:] = mask[:, 1:] * mask[:, :-1]
        valid_gaps = gaps * pair_mask
        max_gap = valid_gaps.amax(dim=1)
        mean_gap = valid_gaps.sum(dim=1) / (pair_mask.sum(dim=1).clamp(min=1))
        v_change = last_v - first_v
        feat = torch.stack(
            [
                obs_count,
                density,
                first_v,
                last_v,
                mean_v,
                v_change,
                max_gap,
                mean_gap,
                (last_t - first_t).clamp(min=0),
                first_t,
                last_t,
                std_v,
                min_v,
                max_v,
                v_change / (last_t - first_t + 1e-8),
            ],
            dim=-1,
        )
        var_ids = torch.arange(channels, device=x.device)
        tokens = self.var_summary_proj(feat) + self.var_emb(var_ids).unsqueeze(0)
        return tokens, torch.ones(bsz, channels, device=x.device), last_t, var_ids.view(1, channels).expand(bsz, channels)

    def forward(self, event_emb, x, t_norm, mask):
        bsz = x.size(0)
        ev_tokens, ev_mask, ev_time, ev_var = self._event_tokens(event_emb, t_norm, mask)
        gap_tokens, gap_mask, gap_time, gap_var = self._gap_tokens(x, t_norm, mask)
        var_tokens, var_mask, var_time, var_var = self._var_tokens(x, t_norm, mask)
        tokens = torch.cat(
            [
                ev_tokens + self.type_emb(torch.full((bsz, self.max_event_tokens), self.TYPE_EVENT, device=x.device)),
                gap_tokens + self.type_emb(torch.full((bsz, self.max_gap_tokens), self.TYPE_GAP, device=x.device)),
                var_tokens + self.type_emb(torch.full((bsz, x.size(-1)), self.TYPE_VAR, device=x.device)),
            ],
            dim=1,
        )
        token_mask = torch.cat([ev_mask, gap_mask, var_mask], dim=1)
        token_time = torch.cat([ev_time, gap_time, var_time], dim=1)
        token_var = torch.cat([ev_var, gap_var, var_var], dim=1)
        return self.norm(self.dropout(tokens)), token_mask, token_time, token_var


class _PatternInteraction(nn.Module):
    def __init__(self, num_channels, d_model, dropout, n_layers, n_heads,
                 max_event_tokens, max_gap_tokens):
        super().__init__()
        self.num_channels = num_channels
        self.max_event_tokens = max_event_tokens
        self.max_gap_tokens = max_gap_tokens
        if n_layers > 0:
            n_heads = max(1, min(n_heads, d_model))
            while d_model % n_heads != 0 and n_heads > 1:
                n_heads -= 1
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.mixer = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.mixer = None
        self.type_proj = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model)) for _ in range(3)]
        )
        self.fusion = nn.Sequential(nn.LayerNorm(d_model * 3), nn.Linear(d_model * 3, d_model), nn.GELU())
        self.global_proj = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, tokens, token_mask):
        if self.mixer is not None:
            tokens = tokens + self.mixer(tokens, src_key_padding_mask=~token_mask.bool())
        event_end = self.max_event_tokens
        gap_end = event_end + self.max_gap_tokens
        if tokens.size(1) != gap_end + self.num_channels:
            raise ValueError("Token sequence length does not match the event, gap, and variable blocks.")
        type_blocks = [
            (tokens[:, :event_end], token_mask[:, :event_end]),
            (tokens[:, event_end:gap_end], token_mask[:, event_end:gap_end]),
            (tokens[:, gap_end:], token_mask[:, gap_end:]),
        ]
        pooled = [
            projection(torch.cat([_masked_mean(block, mask, 1), _masked_max(block, mask, 1)], -1))
            for projection, (block, mask) in zip(self.type_proj, type_blocks)
        ]
        fused = self.fusion(torch.cat(pooled, dim=-1))
        global_mean = _masked_mean(tokens, token_mask.bool(), 1)
        global_max = _masked_max(tokens, token_mask.bool(), 1)
        return self.global_proj(torch.cat([global_mean, global_max, fused], dim=-1))


class ITSPM(TorchBaseModel):
    """Causal local-window ITSPM adapted to marked TPP likelihood training."""

    def __init__(self, model_config):
        super().__init__(model_config)
        specs = model_config.model_specs or {}
        self.d_model = model_config.hidden_size
        self.window_size = int(specs.get('window_size', 32))
        self.max_event_tokens = int(specs.get('max_event_tokens', max(16, self.window_size)))
        self.max_gap_tokens = int(specs.get('max_gap_tokens', max(8, self.window_size // 2)))
        if min(self.window_size, self.max_event_tokens, self.max_gap_tokens) < 1:
            raise ValueError("Window size and token limits must be positive.")
        self.head_type = str(specs.get('head_type', 'decay_mlp'))
        self.time_emb_size = int(specs.get('time_emb_size', self.d_model))
        n_layers = int(specs.get('n_mixer_layers', model_config.num_layers))
        n_heads = int(specs.get('n_heads', model_config.num_heads))
        dropout = model_config.dropout_rate

        self.event_encoder = _EventEncoder(self.num_event_types, self.d_model, dropout)
        self.tokenizer = _SparseTokenizer(
            self.num_event_types,
            self.d_model,
            self.max_event_tokens,
            self.max_gap_tokens,
            dropout,
        )
        self.interaction = _PatternInteraction(
            self.num_event_types, self.d_model, dropout, n_layers, n_heads,
            self.max_event_tokens, self.max_gap_tokens,
        )
        self.sample_time_emb = _SinTimeEmbedding(self.time_emb_size)
        head_in = self.d_model + self.time_emb_size
        if self.head_type == 'linear_decay':
            self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        else:
            self.layer_intensity_hidden = nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Linear(head_in, self.d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model * 2, self.num_event_types),
            )
        self.factor_intensity_base = nn.Parameter(torch.empty(1, self.num_event_types, device=self.device))
        self.factor_intensity_decay = nn.Parameter(torch.empty(1, self.num_event_types, device=self.device))
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)
        self.softplus = ScaledSoftplus(self.num_event_types)

    def _events_to_channels(self, time_seqs, type_seqs):
        bsz, seq_len = type_seqs.shape
        valid = type_seqs.ne(self.pad_token_id)
        safe_types = type_seqs.clamp(min=0, max=self.num_event_types - 1)
        x = F.one_hot(safe_types, num_classes=self.num_event_types).float() * valid.unsqueeze(-1).float()
        mask = x
        t = time_seqs.unsqueeze(-1).expand(bsz, seq_len, self.num_event_types)
        return x, t, mask

    def _build_causal_windows(self, time_seqs, type_seqs):
        bsz, seq_len = type_seqs.shape
        offsets = torch.arange(self.window_size, device=type_seqs.device)
        end_idx = torch.arange(seq_len, device=type_seqs.device).unsqueeze(1)
        idx = end_idx - (self.window_size - 1 - offsets).unsqueeze(0)
        valid = idx >= 0
        idx = idx.clamp(min=0)
        idx = idx.unsqueeze(0).expand(bsz, seq_len, self.window_size)
        win_time = time_seqs.gather(1, idx.reshape(bsz, seq_len * self.window_size)).reshape(bsz, seq_len, self.window_size)
        win_type = type_seqs.gather(1, idx.reshape(bsz, seq_len * self.window_size)).reshape(bsz, seq_len, self.window_size)
        win_type = torch.where(valid.unsqueeze(0), win_type, torch.full_like(win_type, self.pad_token_id))
        return win_time.reshape(bsz * seq_len, self.window_size), win_type.reshape(bsz * seq_len, self.window_size)

    def forward(self, time_seqs, type_seqs, attention_mask=None):
        bsz, seq_len = type_seqs.shape
        win_time, win_type = self._build_causal_windows(time_seqs, type_seqs)
        x, t, mask = self._events_to_channels(win_time, win_type)
        event_emb, t_norm = self.event_encoder(x, t, mask)
        tokens, token_mask, _, _ = self.tokenizer(event_emb, x, t_norm, mask)
        states = self.interaction(tokens, token_mask)
        return states.reshape(bsz, seq_len, self.d_model)

    def compute_states_at_sample_times(self, event_states, sample_dtimes):
        if self.head_type == 'linear_decay':
            base_state = self.layer_intensity_hidden(event_states).unsqueeze(2)
            return base_state + self.factor_intensity_decay.view(1, 1, 1, -1) * sample_dtimes.unsqueeze(-1) + self.factor_intensity_base.view(1, 1, 1, -1)
        time_emb = self.sample_time_emb(sample_dtimes)
        states = event_states.unsqueeze(2).expand(-1, -1, sample_dtimes.size(-1), -1)
        return self.layer_intensity_hidden(torch.cat([states, time_emb], dim=-1)) + self.factor_intensity_base.view(1, 1, 1, -1)

    def loglike_loss(self, batch):
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask = batch
        if time_seqs.size(1) < 2:
            raise ValueError("Likelihood training requires a sequence with at least two events.")
        enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], None)
        event_dtimes = time_delta_seqs[:, 1:]
        lambda_at_event = self.softplus(self.compute_states_at_sample_times(enc_out, event_dtimes.unsqueeze(-1)).squeeze(2))
        sample_dtimes = self.make_dtime_loss_samples(event_dtimes)
        lambda_t_sample = self.softplus(self.compute_states_at_sample_times(enc_out, sample_dtimes))
        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            time_delta_seq=event_dtimes,
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambda_t_sample,
            seq_mask=batch_non_pad_mask[:, 1:] & batch_non_pad_mask[:, :-1],
            type_seq=type_seqs[:, 1:],
        )
        return -(event_ll - non_event_ll).sum(), num_events

    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sample_dtimes, **kwargs):
        compute_last_step_only = kwargs.get('compute_last_step_only', False)
        enc_out = self.forward(time_seqs, type_seqs, None)
        if compute_last_step_only:
            enc_out = enc_out[:, -1:]
            sample_dtimes = sample_dtimes[:, -1:]
        states = self.compute_states_at_sample_times(enc_out, sample_dtimes)
        lambdas = self.softplus(states)
        return lambdas
