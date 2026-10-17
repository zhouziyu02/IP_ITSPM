"""
ITSPM for Irregular Time Series
==================================================================
A lightweight, evidence-aware representation framework for irregular time series.

Tokenization strategy (fully vectorized, no CPU quantile, no .item() in forward):
  - Event tokens      : top-k recent observed events (recency score, vectorized topk)
  - Gap tokens        : top-k largest adjacent observed gaps (vectorized topk)
  - Var summary tokens: one per variable, mask-aware summary statistics

Architecture:
  [1] Event Primitive Encoder  : encode each observation as a rich event embedding
  [2] Fast Sparse Tokenizer    : event / gap / variable-summary tokens — fully batched
  [3] Lightweight Interaction  : type-wise pooling + gated fusion + variable relation MLP
  [4] Task Decoder             : global_repr -> classification; query-token readout -> interp/extrap

Interfaces preserved:
  forward(observed_tp, observed_data, observed_mask, opt=None) -> [B, D]
  forecasting(tp_to_predict, observed_data, observed_tp, observed_mask) -> [1, B, Lp, C]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# 1. Utility Modules
# ─────────────────────────────────────────────────────────────────

class SinTimeEmbedding(nn.Module):
    """Sinusoidal continuous time embedding with linear projection."""
    def __init__(self, d_model: int):
        super().__init__()
        half = d_model // 2
        self.register_buffer('freqs', torch.exp(
            torch.arange(0, half) * -(math.log(10000.0) / max(half - 1, 1))
        ))
        self.proj = nn.Linear(1 + 2 * half, d_model)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_flat = t.unsqueeze(-1)                           # [..., 1]
        angle  = t_flat * self.freqs                        # [..., H]
        pe     = torch.cat([t_flat, torch.sin(angle), torch.cos(angle)], dim=-1)
        return self.proj(pe)


class GatedFusion(nn.Module):
    """Fuses N input vectors via learnable sigmoid gates."""
    def __init__(self, n_inputs: int, d_model: int):
        super().__init__()
        self.gates = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_inputs)])
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, *inputs):
        out = sum(x * torch.sigmoid(g(x)) for x, g in zip(inputs, self.gates)
                  if x is not None)
        return self.norm(out)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean pooling along `dim`, ignoring masked-out positions (mask=0)."""
    mask_f = mask.float().unsqueeze(-1)
    return (x * mask_f).sum(dim) / (mask_f.sum(dim) + 1e-8)


def masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Max pooling along `dim`, ignoring masked-out positions."""
    x_masked = x.masked_fill(~mask.bool().unsqueeze(-1), float('-inf'))
    result, _ = x_masked.max(dim)
    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


# ─────────────────────────────────────────────────────────────────
# 2. Stage 1 — Event Primitive Encoder
# ─────────────────────────────────────────────────────────────────

class EventEncoder(nn.Module):
    """
    Encodes each (time, variable, value, mask, delta_t) observation tuple
    into a D-dimensional event embedding.

    Input shapes (all [B, L, C]):
      X     : observed values
      tp    : timestamps (per-variable if [B,L,C], shared if [B,L])
      mask  : 1 = observed, 0 = missing
    Returns event_emb: [B, L, C, D], t_norm: [B, L, C]
    """

    def __init__(self, num_channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_channels = num_channels

        self.val_proj  = nn.Linear(1, d_model)
        self.var_emb   = nn.Embedding(num_channels, d_model)
        self.time_emb  = SinTimeEmbedding(d_model)
        self.dt_proj   = nn.Linear(1, d_model)
        self.mask_emb  = nn.Embedding(2, d_model)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def compute_time_range(self, tp: torch.Tensor, mask: torch.Tensor):
        """Mask-aware normalization range: only use valid timestamps."""
        valid = mask > 0
        tp_min = tp.masked_fill(~valid, float('inf')).amin(dim=(1, 2), keepdim=True)
        tp_max = tp.masked_fill(~valid, float('-inf')).amax(dim=(1, 2), keepdim=True)
        has_valid = valid.sum(dim=(1, 2), keepdim=True) > 0
        tp_min = torch.where(has_valid, tp_min, tp.amin(dim=(1, 2), keepdim=True))
        tp_max = torch.where(has_valid, tp_max, tp.amax(dim=(1, 2), keepdim=True))
        return tp_min, tp_max

    def encode(self, X: torch.Tensor, tp: torch.Tensor, mask: torch.Tensor,
               tp_min: torch.Tensor, tp_max: torch.Tensor) -> torch.Tensor:
        """
        X, tp, mask: [B, L, C]
        tp_min, tp_max: [B, 1, 1]
        Returns event_emb: [B, L, C, D], t_norm: [B, L, C]
        """
        B, L, C = X.shape

        # Safe normalization — no clamping, just safe denom
        denom = tp_max - tp_min
        denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
        t_norm = (tp - tp_min) / denom   # [B, L, C]

        # Vectorized delta time: time since previous observed event per variable
        obs = mask > 0

        # Use index-based cummax to find the most recent observed position
        seq_idx = torch.arange(1, L + 1, device=tp.device).view(1, L, 1).expand(B, L, C)
        masked_idx = seq_idx.masked_fill(~obs, 0)

        # Shift by 1 to get the strictly previous observation
        prev_masked_idx = torch.cat([torch.zeros_like(masked_idx[:, :1, :]), masked_idx[:, :-1, :]], dim=1)
        last_obs_idx = prev_masked_idx.cummax(dim=1).values

        valid_prev = last_obs_idx > 0
        gather_idx = (last_obs_idx - 1).clamp(min=0)
        last_t = tp.gather(1, gather_idx)

        dt = torch.where(obs & valid_prev, tp - last_t, torch.zeros_like(tp))

        dt_norm = dt / (denom + 1e-8)   # reuse same denom, [B, L, C]

        var_ids = torch.arange(C, device=X.device)         # [C]
        E_val  = self.val_proj(X.unsqueeze(-1))                     # [B, L, C, D]
        E_var  = self.var_emb(var_ids).reshape(1, 1, C, self.d_model)  # [1, 1, C, D]
        E_time = self.time_emb(t_norm)                               # [B, L, C, D]
        E_dt   = self.dt_proj(dt_norm.unsqueeze(-1))                # [B, L, C, D]
        E_mask = self.mask_emb(mask.long())                          # [B, L, C, D]

        E = E_val + E_var + E_time + E_dt + E_mask                  # [B, L, C, D]
        E = self.norm(self.fusion_mlp(E))
        return E, t_norm


# ─────────────────────────────────────────────────────────────────
# 3. Stage 2 — Fast Sparse Pattern Tokenizer (vectorized)
# ─────────────────────────────────────────────────────────────────

class FastSparsePatternTokenizer(nn.Module):
    """
    Fast vectorized tokenizer producing three token types:

      TYPE_EVENT       = 0  top-k recent observed events (recency score)
      TYPE_GAP         = 1  top-k largest adjacent observed gaps
      TYPE_VAR_SUMMARY = 2  one per-variable summary token
      TYPE_PAD         = 3  padding

    Token count:  M = max_event_tokens + max_gap_tokens + C

    Key design:
      - No torch.quantile on CPU
      - No .item() in training forward
      - No nested Python loops over B and C in the main path
      - Variable summary tokens use a small loop over C (cheap)
    """

    TYPE_EVENT       = 0
    TYPE_GAP         = 1
    TYPE_VAR_SUMMARY = 2
    TYPE_PAD         = 3

    def __init__(self, num_channels: int, d_model: int,
                 max_event_tokens: int = 64,
                 max_gap_tokens: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.C       = num_channels
        self.D       = d_model
        self.max_ev  = max_event_tokens
        self.max_gap = max_gap_tokens
        self.M       = max_event_tokens + max_gap_tokens + num_channels

        # Gap token: 6 scalar features
        #   [gap_len, t_left, t_right, v_left, v_right, v_change]
        self.gap_proj = nn.Linear(6, d_model)

        # Variable summary token: 15 scalar features
        #   [count, density, first_v, last_v, mean_v, v_change,
        #    max_gap, mean_gap, time_span, t_first, t_last,
        #    std_v, min_v, max_v, slope]
        self.var_summary_proj = nn.Linear(15, d_model)

        # Type embedding (0=event, 1=gap, 2=var_summary, 3=pad)
        self.type_emb = nn.Embedding(4, d_model)
        # Variable embedding (also used in gap and var-summary tokens)
        self.var_emb  = nn.Embedding(num_channels, d_model)

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Event tokens — fully batched topk
    # ------------------------------------------------------------------

    def _event_tokens(self, event_emb, tp_norm, mask):
        """
        event_emb: [B, L, C, D]
        tp_norm:   [B, L, C]
        mask:      [B, L, C]
        Returns:
          ev_tokens [B, max_ev, D], ev_mask [B, max_ev],
          ev_time [B, max_ev], ev_var [B, max_ev]
        """
        B, L, C, D = event_emb.shape
        device = event_emb.device

        # Flatten time × channel
        ev_flat   = event_emb.reshape(B, L * C, D)      # [B, L*C, D]
        tp_flat   = tp_norm.reshape(B, L * C)            # [B, L*C]
        mask_flat = mask.reshape(B, L * C)               # [B, L*C]

        # Variable index for each flattened position
        var_flat  = torch.arange(C, device=device).unsqueeze(0)   # [1, C]
        var_flat  = var_flat.expand(L, C).contiguous().reshape(-1) # [L*C]
        var_flat  = var_flat.unsqueeze(0).expand(B, L * C)          # [B, L*C]

        # Recency score: observed positions get tp, unobserved get large negative
        score = tp_flat + mask_flat * 1e6 - (1 - mask_flat) * 1e9  # [B, L*C]

        # topk — select max_ev positions per sample
        k = min(self.max_ev, L * C)
        _, idx = torch.topk(score, k, dim=1)           # [B, k]

        ev_emb_sel  = ev_flat.gather(1, idx.unsqueeze(-1).expand(B, k, D))  # [B, k, D]
        ev_time_sel = tp_flat.gather(1, idx)            # [B, k]
        ev_var_sel  = var_flat.gather(1, idx)           # [B, k]
        ev_mask_sel = mask_flat.gather(1, idx)          # [B, k]  1 if observed

        # Pad to max_ev if k < max_ev
        if k < self.max_ev:
            pad = self.max_ev - k
            ev_emb_sel  = F.pad(ev_emb_sel,  (0, 0, 0, pad))
            ev_time_sel = F.pad(ev_time_sel, (0, pad))
            ev_var_sel  = F.pad(ev_var_sel,  (0, pad))
            ev_mask_sel = F.pad(ev_mask_sel, (0, pad))

        return ev_emb_sel, ev_mask_sel.float(), ev_time_sel, ev_var_sel.long()

    # ------------------------------------------------------------------
    # Gap tokens — fully batched topk
    # ------------------------------------------------------------------

    def _gap_tokens(self, X, tp_norm, mask):
        """
        X:       [B, L, C]
        tp_norm: [B, L, C]
        mask:    [B, L, C]
        Returns:
          gap_tokens [B, max_gap, D], gap_mask [B, max_gap],
          gap_time [B, max_gap], gap_var [B, max_gap]
        """
        B, L, C = X.shape
        device  = X.device

        # Compute adjacent gaps; gap at position 0 is 0 (no predecessor)
        gap_len   = torch.zeros_like(tp_norm)            # [B, L, C]
        t_left_g  = torch.zeros_like(tp_norm)
        v_left_g  = torch.zeros_like(X)
        pair_mask = torch.zeros_like(mask)               # 1 if both adjacent observed

        if L > 1:
            gap_len[:, 1:, :]   = (tp_norm[:, 1:, :] - tp_norm[:, :-1, :]).clamp(min=0)
            t_left_g[:, 1:, :]  = tp_norm[:, :-1, :]
            v_left_g[:, 1:, :]  = X[:, :-1, :]
            pair_mask[:, 1:, :] = mask[:, 1:, :] * mask[:, :-1, :]

        # Flatten
        gap_flat  = (gap_len  * pair_mask).reshape(B, L * C)    # [B, L*C]
        tm_flat   = ((tp_norm[:, :, :] + t_left_g[:, :, :]) * 0.5).reshape(B, L * C)  # mid-time
        tl_flat   = t_left_g.reshape(B, L * C)
        tr_flat   = tp_norm.reshape(B, L * C)
        vl_flat   = v_left_g.reshape(B, L * C)
        vr_flat   = X.reshape(B, L * C)
        pm_flat   = pair_mask.reshape(B, L * C)

        var_flat  = torch.arange(C, device=device).unsqueeze(0).expand(L, C).contiguous().reshape(-1)
        var_flat  = var_flat.unsqueeze(0).expand(B, L * C)   # [B, L*C]

        k = min(self.max_gap, L * C)
        _, idx = torch.topk(gap_flat, k, dim=1)               # [B, k]

        # Gather features
        g_len = gap_flat.gather(1, idx)      # [B, k]
        g_tl  = tl_flat.gather(1, idx)
        g_tr  = tr_flat.gather(1, idx)
        g_vl  = vl_flat.gather(1, idx)
        g_vr  = vr_flat.gather(1, idx)
        g_vc  = g_vr - g_vl
        g_tm  = tm_flat.gather(1, idx)
        g_var = var_flat.gather(1, idx)      # [B, k]
        g_pm  = pm_flat.gather(1, idx)       # valid flag

        # Project gap features: [B, k, 6]
        gap_feat = torch.stack([g_len, g_tl, g_tr, g_vl, g_vr, g_vc], dim=-1)
        gap_emb  = self.gap_proj(gap_feat)                   # [B, k, D]
        gap_emb  = gap_emb + self.var_emb(g_var.clamp(min=0, max=self.C - 1))

        # Pad to max_gap if needed
        if k < self.max_gap:
            pad = self.max_gap - k
            gap_emb = F.pad(gap_emb, (0, 0, 0, pad))
            g_pm    = F.pad(g_pm,    (0, pad))
            g_tm    = F.pad(g_tm,    (0, pad))
            g_var   = F.pad(g_var,   (0, pad))

        return gap_emb, g_pm.float(), g_tm, g_var.long()

    # ------------------------------------------------------------------
    # Variable summary tokens — small loop over C
    # ------------------------------------------------------------------

    def _var_summary_tokens(self, X, tp_norm, mask):
        """
        X:       [B, L, C]
        tp_norm: [B, L, C]
        mask:    [B, L, C]
        Returns:
          vs_tokens [B, C, D], vs_mask [B, C],
          vs_time [B, C], vs_var [B, C]
        """
        B, L, C = X.shape
        device  = X.device

        # Vectorized across C
        obs_count   = mask.sum(dim=1)                              # [B, C]
        density     = obs_count / (L + 1e-8)                      # [B, C]

        # Masked min/max time position
        tp_obs_min  = tp_norm.masked_fill(mask == 0, float('inf')).amin(dim=1)   # [B, C]
        tp_obs_max  = tp_norm.masked_fill(mask == 0, float('-inf')).amax(dim=1)  # [B, C]
        # Safe fill: if no obs, set to 0
        has_obs     = obs_count > 0                                # [B, C]
        tp_obs_min  = torch.where(has_obs, tp_obs_min,  torch.zeros_like(tp_obs_min))
        tp_obs_max  = torch.where(has_obs, tp_obs_max,  torch.zeros_like(tp_obs_max))
        time_span   = (tp_obs_max - tp_obs_min).clamp(min=0)      # [B, C]

        # Mean observed value
        mean_v      = (X * mask).sum(dim=1) / (obs_count + 1e-8)  # [B, C]
        centered    = (X - mean_v.unsqueeze(1)) * mask
        std_v       = torch.sqrt((centered * centered).sum(dim=1) / (obs_count + 1e-8) + 1e-8)
        min_v       = X.masked_fill(mask == 0, float('inf')).amin(dim=1)
        max_v       = X.masked_fill(mask == 0, float('-inf')).amax(dim=1)

        # First/last observed value — gather via argmin/argmax of masked positions
        # First observed: argmin of tp_norm among observed
        tp_for_first = tp_norm.masked_fill(mask == 0, float('inf'))
        first_idx    = tp_for_first.argmin(dim=1)                  # [B, C]
        first_v      = X.gather(1, first_idx.unsqueeze(1)).squeeze(1)  # [B, C]
        first_t      = tp_norm.gather(1, first_idx.unsqueeze(1)).squeeze(1)  # [B, C]

        tp_for_last  = tp_norm.masked_fill(mask == 0, float('-inf'))
        last_idx     = tp_for_last.argmax(dim=1)                   # [B, C]
        last_v       = X.gather(1, last_idx.unsqueeze(1)).squeeze(1)   # [B, C]
        last_t       = tp_norm.gather(1, last_idx.unsqueeze(1)).squeeze(1)   # [B, C]

        # Where there are no observations, set first/last to 0
        first_v = torch.where(has_obs, first_v, torch.zeros_like(first_v))
        last_v  = torch.where(has_obs, last_v,  torch.zeros_like(last_v))
        first_t = torch.where(has_obs, first_t, torch.zeros_like(first_t))
        last_t  = torch.where(has_obs, last_t,  torch.zeros_like(last_t))
        min_v   = torch.where(has_obs, min_v,   torch.zeros_like(min_v))
        max_v   = torch.where(has_obs, max_v,   torch.zeros_like(max_v))
        v_change = last_v - first_v                                # [B, C]
        slope = v_change / (last_t - first_t + 1e-8)

        # Max gap and mean gap — compute gaps along time axis
        gaps        = torch.zeros_like(tp_norm)                    # [B, L, C]
        pm          = torch.zeros_like(mask)
        if L > 1:
            gaps[:, 1:, :] = (tp_norm[:, 1:, :] - tp_norm[:, :-1, :]).clamp(min=0)
            pm[:, 1:, :]   = mask[:, 1:, :] * mask[:, :-1, :]
        valid_gaps  = gaps * pm                                    # [B, L, C]
        max_gap     = valid_gaps.amax(dim=1)                       # [B, C]
        gap_count   = pm.sum(dim=1).clamp(min=1)                   # [B, C]
        mean_gap    = valid_gaps.sum(dim=1) / gap_count            # [B, C]

        # Stack 15-dim feature vector: [B, C, 15]
        feat = torch.stack([
            obs_count, density, first_v, last_v, mean_v, v_change,
            max_gap, mean_gap, time_span, first_t, last_t,
            std_v, min_v, max_v, slope
        ], dim=-1)

        vs_emb = self.var_summary_proj(feat)         # [B, C, D]
        var_ids = torch.arange(C, device=device)     # [C]
        vs_emb  = vs_emb + self.var_emb(var_ids).unsqueeze(0)  # [B, C, D]

        vs_mask = torch.ones(B, C, device=device)    # always valid
        vs_time = last_t                             # [B, C]
        vs_var  = var_ids.unsqueeze(0).expand(B, C) # [B, C]

        return vs_emb, vs_mask, vs_time, vs_var

    # ------------------------------------------------------------------
    # Main forward — fully batched
    # ------------------------------------------------------------------

    def forward(self,
                event_emb: torch.Tensor,    # [B, L, C, D]
                X:         torch.Tensor,    # [B, L, C]
                tp_norm:   torch.Tensor,    # [B, L, C] normalized timestamps
                mask:      torch.Tensor,    # [B, L, C]
                ):
        B, L, C, D = event_emb.shape
        device = event_emb.device

        # ── 1. Event tokens ─────────────────────────────────────
        ev_emb, ev_mask, ev_time, ev_var = self._event_tokens(event_emb, tp_norm, mask)
        # [B, max_ev, D], [B, max_ev], [B, max_ev], [B, max_ev]

        # ── 2. Gap tokens ───────────────────────────────────────
        gp_emb, gp_mask, gp_time, gp_var = self._gap_tokens(X, tp_norm, mask)
        # [B, max_gap, D], ...

        # ── 3. Variable summary tokens ───────────────────────────
        vs_emb, vs_mask, vs_time, vs_var = self._var_summary_tokens(X, tp_norm, mask)
        # [B, C, D], ...

        # ── 4. Type embeddings ───────────────────────────────────
        def _type_fill(n, type_id):
            return torch.full((B, n), type_id, device=device, dtype=torch.long)

        ev_type = _type_fill(self.max_ev,  self.TYPE_EVENT)
        gp_type = _type_fill(self.max_gap, self.TYPE_GAP)
        vs_type = _type_fill(C,            self.TYPE_VAR_SUMMARY)

        ev_emb = ev_emb + self.type_emb(ev_type)
        gp_emb = gp_emb + self.type_emb(gp_type)
        vs_emb = vs_emb + self.type_emb(vs_type)

        # ── 5. Concatenate all tokens ────────────────────────────
        tokens     = torch.cat([ev_emb, gp_emb, vs_emb], dim=1)   # [B, M, D]
        token_mask = torch.cat([ev_mask, gp_mask, vs_mask], dim=1) # [B, M]
        token_time = torch.cat([ev_time, gp_time, vs_time], dim=1) # [B, M]
        token_var  = torch.cat([ev_var,  gp_var,  vs_var],  dim=1) # [B, M]
        token_type = torch.cat([ev_type, gp_type, vs_type], dim=1) # [B, M]

        tokens = self.norm(self.dropout(tokens))

        return tokens, token_mask, token_time, token_var, token_type


# ─────────────────────────────────────────────────────────────────
# 4. Stage 3 — Lightweight Pattern Interaction
# ─────────────────────────────────────────────────────────────────

class PatternInteraction(nn.Module):
    """
    Type-wise pooling + gated fusion + variable relation MLP.
    Optional transformer token mixing followed by pooled summaries.

    Token types: 0=event, 1=gap, 2=var_summary

    Produces:
      global_repr [B, D]
      var_repr    [B, C, D]
      tokens      [B, M, D]  (pass-through)
    """

    N_TYPES = 3   # event, gap, var_summary

    def __init__(self, num_channels: int, d_model: int, dropout: float = 0.1,
                 n_layers: int = 2, n_heads: int = 2):
        super().__init__()
        self.C = num_channels
        self.D = d_model

        n_heads = max(1, min(n_heads, d_model))
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        if n_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.token_mixer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        else:
            self.token_mixer = None
        self.mixer_norm = nn.LayerNorm(d_model)

        # Type-wise projections (after mean+max pooling)
        self.type_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model))
            for _ in range(self.N_TYPES)
        ])

        # Gated fusion of type summaries
        self.fusion = GatedFusion(self.N_TYPES, d_model)

        # Variable relation: pool per-variable tokens, then mix across vars
        self.var_pool_proj = nn.Linear(d_model, d_model)
        self.var_rel_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        # Global aggregation
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

    def forward(self,
                tokens:     torch.Tensor,   # [B, M, D]
                token_mask: torch.Tensor,   # [B, M]
                token_type: torch.Tensor,   # [B, M]  0/1/2/3
                token_var:  torch.Tensor,   # [B, M]  0..C-1
                ):
        B, M, D = tokens.shape

        if self.token_mixer is not None:
            mixed = self.token_mixer(tokens, src_key_padding_mask=~token_mask.bool())
            tokens = self.mixer_norm(tokens + mixed)

        # ── Type-wise pooling ──
        type_reprs = []
        for t_id in range(self.N_TYPES):
            t_mask = (token_type == t_id) & token_mask.bool()   # [B, M]
            t_mean = masked_mean(tokens, t_mask, dim=1)          # [B, D]
            t_max  = masked_max(tokens,  t_mask, dim=1)          # [B, D]
            t_pool = torch.cat([t_mean, t_max], dim=-1)          # [B, 2D]
            type_reprs.append(self.type_proj[t_id](t_pool))

        fused = self.fusion(*type_reprs)   # [B, D]

        # ── Variable-wise pooling & relation ──
        # Vectorized: use var_summary tokens (type 2) as var_repr directly
        # They already carry per-variable summaries; just gather and project.
        vs_mask = (token_type == 2) & token_mask.bool()   # [B, M]
        # Expand to [B, C, D]: var_summary tokens are stored contiguously at end
        # (positions max_ev+max_gap .. max_ev+max_gap+C-1)
        # We can safely slice them since tokenizer layout is fixed.
        var_repr = tokens[:, -self.C:, :]                 # [B, C, D]
        var_repr = self.var_pool_proj(var_repr)            # [B, C, D]
        var_repr = var_repr + self.var_rel_mlp(var_repr)  # [B, C, D]

        # ── Global representation ──
        global_mean = masked_mean(tokens, token_mask.bool(), dim=1)   # [B, D]
        global_max  = masked_max(tokens,  token_mask.bool(), dim=1)   # [B, D]
        global_repr = self.global_mlp(
            torch.cat([global_mean, global_max], dim=-1)
        ) + fused                                                      # [B, D]

        return global_repr, var_repr, tokens


class KernelSummaryBranch(nn.Module):
    """Per-variable soft time-kernel summaries, complementary to sparse tokens."""
    def __init__(self, num_channels: int, d_model: int, n_kernels: int = 32, dropout: float = 0.1):
        super().__init__()
        self.C = num_channels
        self.K = n_kernels
        self.register_buffer('centers', torch.linspace(0.0, 1.0, n_kernels))
        self.log_bw = nn.Parameter(torch.full((n_kernels,), -2.2))
        self.var_emb = nn.Embedding(num_channels, d_model)
        self.var_proj = nn.Sequential(
            nn.Linear(n_kernels * 2 + 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, X, t_norm, mask):
        B, L, C = X.shape
        centers = self.centers.view(1, 1, 1, self.K)
        bw = F.softplus(self.log_bw).view(1, 1, 1, self.K) + 1e-4
        dist = t_norm.unsqueeze(-1) - centers
        w = torch.exp(-0.5 * (dist / bw) ** 2) * mask.unsqueeze(-1)
        denom = w.sum(dim=1) + 1e-8
        mean = (w * X.unsqueeze(-1)).sum(dim=1) / denom
        density = denom / (mask.sum(dim=1, keepdim=False).unsqueeze(-1) + 1e-8)

        obs_count = mask.sum(dim=1)
        mean_v = (X * mask).sum(dim=1) / (obs_count + 1e-8)
        last_t = t_norm.masked_fill(mask == 0, float('-inf')).amax(dim=1)
        first_t = t_norm.masked_fill(mask == 0, float('inf')).amin(dim=1)
        has_obs = obs_count > 0
        last_t = torch.where(has_obs, last_t, torch.zeros_like(last_t))
        first_t = torch.where(has_obs, first_t, torch.zeros_like(first_t))
        aux = torch.stack([obs_count / (L + 1e-8), mean_v, first_t, last_t], dim=-1)

        feat = torch.cat([mean, density, aux], dim=-1)
        var_ids = torch.arange(C, device=X.device)
        var_repr = self.var_proj(feat) + self.var_emb(var_ids).unsqueeze(0)
        global_repr = self.global_proj(torch.cat([
            var_repr.mean(dim=1),
            var_repr.max(dim=1).values,
        ], dim=-1))
        return global_repr, var_repr


# ─────────────────────────────────────────────────────────────────
# 5. Stage 4 — Task Decoders
# ─────────────────────────────────────────────────────────────────

class QueryTokenReadout(nn.Module):
    """
    Query-driven readout from sparse pattern tokens.

    Given a query (t_q, var_id), computes attention-weighted sum over tokens
    using:
      - learned similarity  q · W · token
      - time proximity bias
      - same-variable bias

    Output: [B, Lp, C]
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.D = d_model

        self.time_emb = SinTimeEmbedding(d_model)
        self.q_proj          = nn.Linear(d_model, d_model)
        self.k_proj          = nn.Linear(d_model, d_model)
        self.time_bias_proj  = nn.Linear(1, 1, bias=False)
        self.var_bias        = nn.Parameter(torch.tensor(0.5))

        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self,
                query_time:  torch.Tensor,   # [B, Lp, C]
                query_var:   torch.Tensor,   # [C]
                global_repr: torch.Tensor,   # [B, D]
                var_repr:    torch.Tensor,   # [B, C, D]
                tokens:      torch.Tensor,   # [B, M, D]
                token_mask:  torch.Tensor,   # [B, M]
                token_time:  torch.Tensor,   # [B, M]
                token_var:   torch.Tensor,   # [B, M]  long
                ) -> torch.Tensor:
        B, Lp, C = query_time.shape
        M = tokens.shape[1]
        D = self.D

        # Query embedding from variable state plus target time.
        q_var_emb = var_repr.unsqueeze(1).expand(B, Lp, C, D)   # [B, Lp, C, D]
        q_time_emb = self.time_emb(query_time)                    # [B, Lp, C, D]
        Q = self.q_proj(q_var_emb + q_time_emb)                   # [B, Lp, C, D]

        K = self.k_proj(tokens)                                   # [B, M, D]

        # Score: [B, Lp, C, M]
        score = torch.einsum('blcd,bmd->blcm', Q, K)

        # Time distance bias
        dist = torch.abs(query_time.unsqueeze(-1) - token_time.unsqueeze(1).unsqueeze(1))  # [B, Lp, C, M]
        score = score - self.time_bias_proj(dist.unsqueeze(-1)).squeeze(-1)

        # Same-variable bonus
        same_var = (query_var.view(1, 1, C, 1) == token_var.view(B, 1, 1, M)).float()
        score = score + self.var_bias * same_var

        # Mask padding
        pad_mask = (~token_mask.bool()).reshape(B, 1, 1, M)
        score = score.masked_fill(pad_mask, float('-inf'))
        attn  = torch.softmax(score, dim=-1)
        attn  = torch.nan_to_num(attn, nan=0.0)

        # Weighted sum
        ctx = torch.einsum('blcm,bmd->blcd', attn, tokens)      # [B, Lp, C, D]

        g_exp   = global_repr.reshape(B, 1, 1, D).expand(B, Lp, C, D)
        dec_in  = torch.cat([ctx, g_exp, q_var_emb, q_time_emb], dim=-1)
        y       = self.decoder(dec_in).squeeze(-1)               # [B, Lp, C]
        return y


# ─────────────────────────────────────────────────────────────────
# 6. Main Model: ITSPM
# ─────────────────────────────────────────────────────────────────

class ITSPM(nn.Module):
    """
    ITSPM: a backbone for irregular time series.

    Replaces slow dynamic segment/silence tokenizer with fully vectorized:
      - event tokens   (top-k recent events)
      - gap tokens     (top-k largest gaps)
      - var summary    (per-variable statistics)

    Supports three tasks:
      - Classification : forward() -> [B, D]   (external Classifier head)
      - Interpolation  : forecasting() -> [1, B, Lp, C]
      - Extrapolation  : forecasting() -> [1, B, Lp, C]
    """

    def __init__(self, args):
        super().__init__()
        self.d_model      = getattr(args, 'd_model', 64)
        self.num_channels = getattr(args, 'input_dim', 36)
        self.dropout_p    = getattr(args, 'dropout', 0.1)

        n_ref = getattr(args, 'n_ref_points', 32)
        max_event_tokens = getattr(args, 'max_event_tokens', None)
        max_gap_tokens = getattr(args, 'max_gap_tokens', None)
        self.max_event_tokens = max_event_tokens if max_event_tokens is not None else max(32, 2 * n_ref)
        self.max_gap_tokens   = max_gap_tokens if max_gap_tokens is not None else max(16, n_ref)
        self.M = self.max_event_tokens + self.max_gap_tokens + self.num_channels

        D = self.d_model
        C = self.num_channels

        # Stage 1
        self.encoder = EventEncoder(C, D, self.dropout_p)

        # Stage 2
        self.tokenizer = FastSparsePatternTokenizer(
            num_channels    = C,
            d_model         = D,
            max_event_tokens = self.max_event_tokens,
            max_gap_tokens   = self.max_gap_tokens,
            dropout          = self.dropout_p,
        )

        # Stage 3
        self.interaction = PatternInteraction(
            C, D, self.dropout_p,
            n_layers=getattr(args, 'n_mixer_layers', 2),
            n_heads=getattr(args, 'n_scales', 2),
        )
        self.kernel_branch = KernelSummaryBranch(
            C, D,
            n_kernels=min(48, max(16, n_ref)),
            dropout=self.dropout_p,
        )

        # Stage 4 decoders
        self.readout = QueryTokenReadout(D, self.dropout_p)
        self.local_log_bw = nn.Parameter(torch.tensor([-3.0, -1.6, -0.3]))
        self.local_scale = nn.Parameter(torch.tensor(1.0))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.cls_fusion = nn.Sequential(
            nn.LayerNorm(D * 3),
            nn.Linear(D * 3, D * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(D * 2, D),
        )

        self.register_buffer('var_ids', torch.arange(C))

    def _prepare_tp(self, tp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Expand tp to [B, L, C] if it is [B, L]."""
        B, L, C = mask.shape
        if tp.dim() not in (2, 3) or tp.shape[:2] != (B, L):
            raise ValueError('Timestamps must have shape [B, L], [B, L, 1], or [B, L, C].')
        if tp.dim() == 2:
            tp = tp.unsqueeze(-1).expand(B, L, C)
        elif tp.dim() == 3 and tp.size(-1) == 1 and C > 1:
            tp = tp.expand(B, L, C)
        if tp.shape != mask.shape:
            raise ValueError('The timestamp channel count must match the observation channels.')
        return tp

    def _encode(self, X, tp, mask):
        """Full encode -> pattern tokens."""
        if X.dim() != 3 or X.shape != mask.shape or X.shape[1] == 0:
            raise ValueError('Data and mask must have the same nonempty shape [B, L, C].')
        if X.shape[2] != self.num_channels:
            raise ValueError('The input channel count does not match input_dim.')
        tp = self._prepare_tp(tp, mask)
        tp_min, tp_max = self.encoder.compute_time_range(tp, mask)
        event_emb, t_norm = self.encoder.encode(X, tp, mask, tp_min, tp_max)
        tokens, token_mask, token_time, token_var, token_type = self.tokenizer(
            event_emb, X, t_norm, mask)
        global_repr, var_repr, tokens = self.interaction(
            tokens, token_mask, token_type, token_var)
        kernel_global, kernel_var = self.kernel_branch(X, t_norm, mask)
        global_repr = global_repr + kernel_global
        var_repr = var_repr + kernel_var
        return (global_repr, var_repr, tokens, token_mask, token_time,
                token_var, token_type, tp_min, tp_max)

    def _local_kernel_readout(self, query_time, X, t_norm, mask):
        """Same-variable multi-bandwidth kernel smoother used as a regression prior."""
        dist = torch.abs(query_time.unsqueeze(2) - t_norm.unsqueeze(1))
        bw = F.softplus(self.local_log_bw).view(1, 1, 1, 1, -1) + 1e-4
        w = torch.exp(-0.5 * (dist.unsqueeze(-1) / bw) ** 2) * mask.unsqueeze(1).unsqueeze(-1)
        y = (w * X.unsqueeze(1).unsqueeze(-1)).sum(dim=2) / (w.sum(dim=2) + 1e-8)
        return y.mean(dim=-1)

    def forward(self, observed_tp, observed_data, observed_mask, opt=None):
        """
        Classification forward.
        observed_tp:   [B, L] or [B, L, C]
        observed_data: [B, L, C]
        observed_mask: [B, L, C]
        Returns: [B, D]  (fed into external Classifier head)
        """
        global_repr, var_repr, *_ = self._encode(observed_data, observed_tp, observed_mask)
        if opt is not None and getattr(opt, 'disable_cls_fusion', False):
            return global_repr
        var_mean = var_repr.mean(dim=1)
        var_max = var_repr.max(dim=1).values
        cls_repr = self.cls_fusion(torch.cat([global_repr, var_mean, var_max], dim=-1))
        return global_repr + cls_repr   # [B, D]

    def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask=None):
        """
        Interpolation / Extrapolation forward.

        tp_to_predict:  [B, Lp]     normalized target times (shared across vars)
        observed_data:  [B, L, C]
        observed_tp:    [B, L] or [B, L, C]
        observed_mask:  [B, L, C]

        Returns: [1, B, Lp, C]
        """
        B, L, C = observed_data.shape
        if tp_to_predict.dim() not in (2, 3) or tp_to_predict.shape[0] != B:
            raise ValueError('Prediction timestamps must include the same batch dimension as observations.')
        if observed_mask is None:
            observed_mask = torch.ones_like(observed_data)

        (global_repr, var_repr, tokens, token_mask,
         token_time, token_var, token_type, tp_min, tp_max) = self._encode(
            observed_data, observed_tp, observed_mask)
        observed_tp_full = self._prepare_tp(observed_tp, observed_mask)

        # Normalize prediction times
        denom = tp_max - tp_min
        denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
        if tp_to_predict.dim() == 2:
            Lp = tp_to_predict.shape[1]
            tp_pred_norm = (tp_to_predict.unsqueeze(-1) - tp_min) / denom
            tp_pred_norm = tp_pred_norm.expand(B, Lp, C)
        elif tp_to_predict.dim() == 3 and tp_to_predict.shape[0] == B and tp_to_predict.shape[2] in (1, C):
            Lp = tp_to_predict.shape[1]
            tp_pred_norm = (tp_to_predict - tp_min) / denom
            tp_pred_norm = tp_pred_norm.expand(B, Lp, C)
        else:
            raise ValueError('Prediction timestamps must have shape [B, Lp], [B, Lp, 1], or [B, Lp, C].')

        residual = self.readout(
            query_time   = tp_pred_norm,   # [B, Lp, C]
            query_var    = self.var_ids,   # [C]
            global_repr  = global_repr,    # [B, D]
            var_repr     = var_repr,       # [B, C, D]
            tokens       = tokens,         # [B, M, D]
            token_mask   = token_mask,     # [B, M]
            token_time   = token_time,     # [B, M]
            token_var    = token_var,      # [B, M]
        )
        t_norm = (observed_tp_full - tp_min) / denom
        local = self._local_kernel_readout(tp_pred_norm, observed_data, t_norm, observed_mask)
        y = self.local_scale * local + self.residual_scale * residual
        return y.unsqueeze(0)   # [1, B, Lp, C]
