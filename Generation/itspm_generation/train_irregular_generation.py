"""Observed-entry autoencoding followed by unconditional latent diffusion.

The generator is trained in two stages. Evaluation sequences are used for
scoring, not as conditioning inputs to the sampler.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.ITSPM import ITSPM  # noqa: E402


DIFF_MN_TABLE1 = {
    "ds": {
        0.3: {"sines": 0.105, "stocks": 0.142, "energy": 0.422, "mujoco": 0.293},
        0.5: {"sines": 0.128, "stocks": 0.137, "energy": 0.487, "mujoco": 0.375},
        0.7: {"sines": 0.182, "stocks": 0.106, "energy": 0.497, "mujoco": 0.393},
    },
    "mdd": {
        0.3: {"sines": 0.953, "stocks": 0.25, "energy": 0.270, "mujoco": 0.347},
        0.5: {"sines": 1.093, "stocks": 0.281, "energy": 0.252, "mujoco": 0.318},
        0.7: {"sines": 1.308, "stocks": 0.299, "energy": 0.279, "mujoco": 0.297},
    },
    "kl": {
        0.3: {"sines": 0.013, "stocks": 0.074, "energy": 0.020, "mujoco": 0.021},
        0.5: {"sines": 0.023, "stocks": 0.094, "energy": 0.022, "mujoco": 0.009},
        0.7: {"sines": 0.033, "stocks": 0.091, "energy": 0.017, "mujoco": 0.014},
    },
}


@dataclass
class ITSPMConfig:
    input_dim: int
    d_model: int = 64
    dropout: float = 0.1
    n_ref_points: int = 24
    n_scales: int = 2
    n_mixer_layers: int = 1
    max_event_tokens: int | None = None
    max_gap_tokens: int | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sine_dataset(n_samples: int, seq_len: int, channels: int, seed: int) -> np.ndarray:
    # Matches Diff-MN/TimeGAN style Sines: frequency and phase in [0, 0.1],
    # normalized to [0, 1].
    rng = np.random.default_rng(seed)
    data = []
    for i in range(n_samples):
        temp = []
        for c in range(channels):
            freq = rng.uniform(0.0, 0.1)
            phase = rng.uniform(0.0, 0.1)
            temp.append([np.sin(freq * j + phase) for j in range(seq_len)])
        data.append(((np.asarray(temp).T + 1.0) * 0.5).astype(np.float32))
    return np.asarray(data, dtype=np.float32)


def make_polynomial_dataset(n_samples: int, seq_len: int, channels: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(-1.0, 1.0, seq_len, dtype=np.float32)
    powers = np.stack([t, t**2, t**3], axis=1)
    data = np.zeros((n_samples, seq_len, channels), dtype=np.float32)
    for i in range(n_samples):
        shared = rng.normal(0.0, 0.45, size=(3,))
        for c in range(channels):
            coeff = shared + rng.normal(0.0, 0.2, size=(3,))
            data[i, :, c] = powers @ coeff + rng.normal(0.0, 0.03, size=seq_len)
    return data


def minmax_scale(data: np.ndarray) -> np.ndarray:
    data_min = data.min(axis=0, keepdims=True)
    data_max = data.max(axis=0, keepdims=True)
    return (data - data_min) / (data_max - data_min + 1e-7)


def load_windowed_csv(path: Path, seq_len: int, n_samples: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing real dataset CSV: {path}. Place the corresponding Diff-MN "
            "stock_data.csv or energy_data.csv under --data_root. "
            "No synthetic substitute is generated for Stocks or Energy."
        )
    try:
        raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2).astype(np.float32)
    except ValueError as exc:
        raise ValueError(f"{path}: expected numeric CSV columns after one header row.") from exc
    if raw.shape[0] < seq_len or raw.shape[1] == 0:
        raise ValueError(f"{path}: need at least {seq_len} rows and one numeric channel; got {raw.shape}.")
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: dataset contains NaN or infinity.")
    raw = minmax_scale(raw[::-1])
    windows = np.stack([raw[i:i + seq_len] for i in range(0, len(raw) - seq_len + 1)], axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(windows))
    windows = windows[idx]
    if n_samples is not None and n_samples > 0:
        windows = windows[:n_samples]
        idx = idx[:n_samples]
    return windows.astype(np.float32), idx


def make_sample_timestep_mask(shape: tuple[int, int, int], missing: float, seed: int = 56789) -> np.ndarray:
    # Official Diff-MN loader drops full timesteps per sample for Sines/MuJoCo.
    n, seq_len, channels = shape
    mask = np.ones(shape, dtype=np.float32)
    n_drop = int(seq_len * missing)
    generator = torch.Generator().manual_seed(seed)
    for i in range(n):
        if n_drop > 0:
            removed = torch.randperm(seq_len, generator=generator)[:n_drop].numpy()
            mask[i, removed, :] = 0.0
    mask[:, 0, :] = 1.0
    mask[:, -1, :] = 1.0
    return mask


def make_global_window_mask(total_len: int, channels: int, seq_len: int, missing: float, idx: np.ndarray) -> np.ndarray:
    # Official Diff-MN loader drops full rows on the raw continuous time axis
    # for stock/energy, then applies the same shuffled window order.
    full_mask = np.ones((total_len, channels), dtype=np.float32)
    n_drop = int(total_len * missing)
    if n_drop > 0:
        generator = torch.Generator().manual_seed(56789)
        removed = torch.randperm(total_len, generator=generator)[:n_drop].numpy()
        full_mask[removed, :] = 0.0
    windows = np.stack([full_mask[i:i + seq_len] for i in range(0, total_len - seq_len + 1)], axis=0)
    return windows[idx].astype(np.float32)


def load_table1_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, bool]:
    dataset = "sines" if args.dataset == "sine" else args.dataset
    if dataset == "sines":
        data = make_sine_dataset(args.n_samples, args.seq_len, args.channels, args.seed)
        return data, make_sample_timestep_mask(data.shape, args.missing), True
    if dataset == "stocks":
        path = Path(args.data_root) / "stock_data.csv"
        data, idx = load_windowed_csv(path, args.seq_len, args.n_samples, args.seed)
        raw = np.loadtxt(path, delimiter=",", skiprows=1).astype(np.float32)
        mask = make_global_window_mask(len(raw), data.shape[-1], args.seq_len, args.missing, idx)
        return data, mask, True
    if dataset == "energy":
        path = Path(args.data_root) / "energy_data.csv"
        data, idx = load_windowed_csv(path, args.seq_len, args.n_samples, args.seed)
        raw = np.loadtxt(path, delimiter=",", skiprows=1).astype(np.float32)
        mask = make_global_window_mask(len(raw), data.shape[-1], args.seq_len, args.missing, idx)
        return data, mask, True
    if dataset == "mujoco":
        pt = Path(args.data_root) / f"mujoco_training_{args.seq_len}.pt"
        if not pt.is_file():
            raise FileNotFoundError(
                f"Missing real MuJoCo trajectory file: {pt}. "
                "From the repository root, install the physics dependencies with "
                "'python Generation/scripts/install_physics.py', then run "
                f"'python Generation/scripts/prepare_mujoco.py --lengths {args.seq_len}', "
                "or provide the matching trajectory file under --data_root. "
                "Synthetic proxy fallback is disabled."
            )
        tensor = torch.load(pt, map_location="cpu", weights_only=True)
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 3
                or tuple(tensor.shape[1:]) != (args.seq_len, 14) or tensor.shape[0] == 0):
            raise ValueError(f"{pt}: expected a nonempty MuJoCo tensor shaped [N,{args.seq_len},14].")
        data = tensor.detach().cpu().numpy()
        if not np.isfinite(data).all():
            raise ValueError(f"{pt}: trajectory data contains NaN or infinity.")
        if args.n_samples > 0:
            data = data[:args.n_samples]
        data = minmax_scale(data.reshape(-1, data.shape[-1])).reshape(data.shape).astype(np.float32)
        return data, make_sample_timestep_mask(data.shape, args.missing), True
    if dataset == "polynomial":
        data = make_polynomial_dataset(args.n_samples, args.seq_len, args.channels, args.seed)
        return data, make_sample_timestep_mask(data.shape, args.missing), False
    raise ValueError(f"Unknown dataset: {args.dataset}")


def normalize_train_test(
    data: np.ndarray, train_frac: float, mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the additional z-score transform on observed training entries only.

    Dataset loading and benchmark min-max preprocessing are kept separate.
    Complete reference values are transformed for scoring, never used to fit
    these statistics or to supervise missing-entry reconstruction.
    """
    if data.shape != mask.shape or data.ndim != 3:
        raise ValueError("data and mask must have the same [N,T,C] shape")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must leave both training and evaluation sequences")
    split = int(len(data) * train_frac)
    if split == 0 or split == len(data):
        raise ValueError("The split must contain both training and evaluation sequences")
    observed = mask[:split] > 0
    count = observed.sum(axis=(0, 1), keepdims=True)
    if np.any(count == 0):
        raise ValueError("Each channel needs at least one observed training value")
    values = np.where(observed, data[:split], 0.0).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Observed training values must be finite")
    mean = values.sum(axis=(0, 1), keepdims=True) / count
    centered = np.where(observed, values - mean, 0.0)
    variance = np.square(centered).sum(axis=(0, 1), keepdims=True) / count
    mean = mean.astype(np.float32)
    std = (np.sqrt(variance) + 1e-6).astype(np.float32)
    return (data - mean) / std, mean, std, np.arange(split)


def denormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (data * std + mean).astype(np.float32)


class SinTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.arange(half) * -(math.log(10000.0) / max(half - 1, 1)))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(1 + 2 * half, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angle = t.unsqueeze(-1) * self.freqs
        return self.proj(torch.cat([t.unsqueeze(-1), torch.sin(angle), torch.cos(angle)], dim=-1))


class ITSPMLatentAutoencoder(nn.Module):
    """Encode irregular training observations; decode latents at query times."""
    def __init__(self, cfg: ITSPMConfig, seq_len: int, channels: int, latent_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.backbone = ITSPM(cfg)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, latent_dim),
        )
        self.time_emb = SinTimeEmbedding(latent_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, channels),
        )

    def encode(self, times: torch.Tensor, x_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.backbone(times, x_obs, mask)
        return self.to_latent(h)

    def decode(self, z: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = self.seq_len
        bsz = z.size(0)
        t = torch.linspace(0.0, 1.0, seq_len, device=z.device).unsqueeze(0).expand(bsz, seq_len)
        te = self.time_emb(t)
        zt = z.unsqueeze(1).expand(bsz, seq_len, z.size(-1))
        return self.decoder(torch.cat([zt, te], dim=-1))

    def forward(self, times: torch.Tensor, x_obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(times, x_obs, mask)
        return self.decode(z), z


class LatentDenoiser(nn.Module):
    """Noise predictor with no observation- or mask-conditioning interface."""
    def __init__(self, latent_dim: int, hidden: int, n_steps: int):
        super().__init__()
        self.time_emb = nn.Embedding(n_steps, hidden)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, self.time_emb(t)], dim=-1))


class LatentDDPM:
    def __init__(self, n_steps: int, beta_start: float, beta_end: float, device: torch.device):
        self.n_steps = n_steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t].view(-1, *([1] * (z0.dim() - 1)))
        return torch.sqrt(ab) * z0 + torch.sqrt(1.0 - ab) * noise

    def predict_x0(self, z_t: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t].view(-1, *([1] * (z_t.dim() - 1)))
        return (z_t - torch.sqrt(1.0 - ab) * pred_noise) / (torch.sqrt(ab) + 1e-8)

    @torch.no_grad()
    def sample(self, denoiser: LatentDenoiser, n_samples: int, latent_dim: int) -> torch.Tensor:
        z = torch.randn(n_samples, latent_dim, device=self.device)
        for step in reversed(range(self.n_steps)):
            t = torch.full((n_samples,), step, dtype=torch.long, device=self.device)
            eps = denoiser(z, t)
            beta = self.betas[step]
            alpha = self.alphas[step]
            ab = self.alpha_bar[step]
            z = (z - beta / torch.sqrt(1.0 - ab) * eps) / torch.sqrt(alpha)
            if step > 0:
                z = z + torch.sqrt(beta) * torch.randn_like(z)
        return z


def observed_reconstruction_loss(
    reconstruction: torch.Tensor, observed_values: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Reconstruction MSE evaluated strictly at observed training entries."""
    if reconstruction.shape != observed_values.shape or mask.shape != reconstruction.shape:
        raise ValueError("Reconstruction, observations, and mask must share a shape")
    observed = mask > 0
    if not observed.any():
        raise ValueError("Reconstruction requires at least one observed entry")
    # Select before subtraction: unobserved values (including NaNs) never enter
    # the loss, and the decoder receives no gradient at those positions.
    return F.mse_loss(reconstruction[observed], observed_values[observed])


def freeze_autoencoder(autoencoder: ITSPMLatentAutoencoder) -> None:
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    for parameter in autoencoder.parameters():
        parameter.grad = None


def standardize_training_latents(
    latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if latents.ndim != 2 or len(latents) == 0 or not torch.isfinite(latents).all():
        raise ValueError("Expected a nonempty, finite [N,latent_dim] training tensor")
    mean = latents.mean(dim=0, keepdim=True)
    # Population statistics stay finite even for a one-sequence training set.
    std = latents.std(dim=0, keepdim=True, unbiased=False) + 1e-6
    return (latents - mean) / std, mean, std


@torch.no_grad()
def sample_sequences(
    autoencoder: ITSPMLatentAutoencoder,
    denoiser: LatentDenoiser,
    diffusion: LatentDDPM,
    n_samples: int,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    seq_len: int | None = None,
) -> torch.Tensor:
    """Generate in normalized data space using noise and learned parameters.

    The optional sequence length defines the regular output-time grid. This
    interface deliberately accepts no reference observations or masks.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    autoencoder.eval()
    denoiser.eval()
    z = diffusion.sample(denoiser, n_samples, latent_mean.shape[-1])
    z = z * latent_std + latent_mean
    return autoencoder.decode(z, seq_len=seq_len)


def save_generation_checkpoint(
    path: Path, autoencoder: ITSPMLatentAutoencoder, denoiser: LatentDenoiser,
    cfg: ITSPMConfig, args: argparse.Namespace, mean: np.ndarray, std: np.ndarray,
    latent_mean: torch.Tensor, latent_std: torch.Tensor,
) -> None:
    """Save everything needed to sample without loading a reference dataset."""
    torch.save({
        "format_version": 1,
        "generator": "itspm_latent_diffusion",
        "autoencoder": autoencoder.state_dict(),
        "denoiser": denoiser.state_dict(),
        "itspm_config": asdict(cfg),
        "args": vars(args),
        "normalization": {
            "data_mean": torch.from_numpy(mean.copy()),
            "data_std": torch.from_numpy(std.copy()),
            "latent_mean": latent_mean.detach().cpu(),
            "latent_std": latent_std.detach().cpu(),
        },
    }, path)


@torch.no_grad()
def sample_from_checkpoint(
    path: Path, n_samples: int, device: torch.device, seed: int | None = None,
) -> np.ndarray:
    """Sample directly from a latent-head checkpoint; no data loader is used."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (checkpoint.get("format_version") != 1
            or checkpoint.get("generator") != "itspm_latent_diffusion"):
        raise ValueError("Expected a latent-head checkpoint with normalization statistics")
    cfg = ITSPMConfig(**checkpoint["itspm_config"])
    args = argparse.Namespace(**checkpoint["args"])
    autoencoder = ITSPMLatentAutoencoder(cfg, args.seq_len, args.channels, args.latent_dim).to(device)
    autoencoder.load_state_dict(checkpoint["autoencoder"])
    freeze_autoencoder(autoencoder)
    denoiser = LatentDenoiser(args.latent_dim, args.diffusion_hidden, args.diffusion_steps).to(device)
    denoiser.load_state_dict(checkpoint["denoiser"])
    diffusion = LatentDDPM(args.diffusion_steps, args.beta_start, args.beta_end, device)
    if seed is not None:
        set_seed(seed)
    stats = checkpoint["normalization"]
    generated = sample_sequences(
        autoencoder, denoiser, diffusion, n_samples,
        stats["latent_mean"].to(device), stats["latent_std"].to(device),
    )
    generated = generated * stats["data_std"].to(device) + stats["data_mean"].to(device)
    return generated.cpu().numpy().astype(np.float32)


def rbf_mmd(x: np.ndarray, y: np.ndarray, max_samples: int = 512) -> float:
    rng = np.random.default_rng(0)
    if len(x) > max_samples:
        x = x[rng.choice(len(x), max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), max_samples, replace=False)]
    x = x.reshape(len(x), -1).astype(np.float64, copy=False)
    y = y.reshape(len(y), -1).astype(np.float64, copy=False)
    xy = np.vstack([x, y])
    # Keep the distance matrix two-dimensional rather than materializing
    # [samples, samples, sequence_length * channels] temporary arrays.
    squared_norm = np.square(xy).sum(axis=1)
    sq = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (xy @ xy.T)
    np.maximum(sq, 0.0, out=sq)
    np.fill_diagonal(sq, 0.0)
    positive = sq[sq > 0]
    bandwidth = np.median(positive) if positive.size else 0.0
    gamma = 1.0 / (2.0 * (bandwidth + 1e-6))
    kernel = np.exp(-gamma * sq)
    n = len(x)
    kxx = kernel[:n, :n].mean()
    kyy = kernel[n:, n:].mean()
    kxy = kernel[:n, n:].mean()
    return float(kxx + kyy - 2.0 * kxy)


def flat_kl(real: np.ndarray, fake: np.ndarray) -> float:
    hist_real, edge_real = np.histogram(real[~np.isnan(real)], density=True, bins=50)
    counts_fake, _ = np.histogram(fake[~np.isnan(fake)], bins=edge_real)
    # NumPy density=True divides by zero if every generated value falls
    # outside the reference range. Preserve zero counts for epsilon smoothing.
    hist_fake = counts_fake / max(counts_fake.sum(), 1) / np.diff(edge_real)
    return float(entropy(hist_real, hist_fake + 1e-9))


def official_mdd(real: np.ndarray, fake: np.ndarray, n_bins: int = 20) -> float:
    x_real = torch.as_tensor(real, dtype=torch.float32)
    x_fake = torch.as_tensor(fake, dtype=torch.float32)
    losses = []
    for c in range(x_real.shape[2]):
        for t in range(x_real.shape[1]):
            xr = x_real[:, t, c]
            xf = x_fake[:, t, c]
            a = xr.min().item()
            b = xr.max().item()
            if b == a:
                b += 1e-2
            bins = torch.linspace(a, b, n_bins + 1)
            delta = bins[1] - bins[0]
            real_density = torch.histc(xr, bins=n_bins, min=a, max=b).float()
            fake_density = torch.histc(xf, bins=n_bins, min=a, max=b).float()
            real_density = real_density / delta / float(xr.numel())
            fake_density = fake_density / delta / float(xf.numel())
            losses.append(torch.abs(fake_density - real_density).mean())
    return float(torch.stack(losses).mean().item())


def acf_error(real: np.ndarray, fake: np.ndarray, max_lag: int = 8) -> float:
    max_lag = min(max_lag, real.shape[1] - 1, fake.shape[1] - 1)
    if max_lag < 1:
        raise ValueError("ACF scoring requires at least two timesteps and a positive max_lag")
    errs = []
    for lag in range(1, max_lag + 1):
        r = (real[:, :-lag] * real[:, lag:]).mean(axis=(0, 1))
        f = (fake[:, :-lag] * fake[:, lag:]).mean(axis=(0, 1))
        errs.append(np.abs(r - f).mean())
    return float(np.mean(errs))


def table1_discriminative_score(
    real: np.ndarray,
    fake: np.ndarray,
    device: torch.device,
    iterations: int,
    batch_size: int,
    hidden_dim: int | None = None,
) -> float:
    if min(len(real), len(fake)) < 2:
        raise ValueError("Discriminative scoring requires at least two real and two generated sequences")
    real_t = torch.as_tensor(real, dtype=torch.float32, device=device)
    fake_t = torch.as_tensor(fake, dtype=torch.float32, device=device)
    if hidden_dim is None:
        hidden_dim = max(4, real.shape[-1] // 2)

    class RNNDiscriminator(nn.Module):
        def __init__(self, inp_dim: int, hidden: int):
            super().__init__()
            self.rnn = nn.GRU(inp_dim, hidden, batch_first=True)
            self.linear = nn.Linear(hidden, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, h = self.rnn(x)
            return self.linear(h[-1]).squeeze(-1)

    rng = np.random.default_rng(0)
    idx_real = rng.permutation(len(real))
    idx_fake = rng.permutation(len(fake))
    split_real = int(0.8 * len(real))
    split_fake = int(0.8 * len(fake))
    train_real = torch.as_tensor(idx_real[:split_real], dtype=torch.long, device=device)
    test_real = torch.as_tensor(idx_real[split_real:], dtype=torch.long, device=device)
    train_fake = torch.as_tensor(idx_fake[:split_fake], dtype=torch.long, device=device)
    test_fake = torch.as_tensor(idx_fake[split_fake:], dtype=torch.long, device=device)
    model = RNNDiscriminator(real.shape[-1], hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters())
    for _ in range(iterations):
        ridx = train_real[torch.randint(0, len(train_real), (batch_size,), device=device)]
        fidx = train_fake[torch.randint(0, len(train_fake), (batch_size,), device=device)]
        logits_real = model(real_t[ridx])
        logits_fake = model(fake_t[fidx])
        loss = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
        loss = loss + F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(real_t[test_real]), model(fake_t[test_fake])], dim=0)
        labels = torch.cat([torch.ones(len(test_real), device=device), torch.zeros(len(test_fake), device=device)])
        pred = (torch.sigmoid(logits) > 0.5).float()
        acc = (pred == labels).float().mean().item()
    return float(abs(0.5 - acc))


def table1_metrics(real: np.ndarray, fake: np.ndarray, device: torch.device, args: argparse.Namespace) -> dict:
    ds = table1_discriminative_score(real, fake, device, args.ds_iterations, args.ds_batch_size)
    return {
        "ds": ds,
        "mdd": official_mdd(real, fake, args.mdd_bins),
        "kl": flat_kl(real, fake),
        "mmd": rbf_mmd(real, fake),
        "acf_error": acf_error(real, fake),
    }


def baseline_gaussian(train: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = train.reshape(len(train), -1)
    mu = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-4
    return rng.normal(mu, std, size=(n_samples, flat.shape[1])).astype(np.float32).reshape(n_samples, *train.shape[1:])


def iter_batches(n: int, batch_size: int, shuffle: bool, seed: int):
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def train(args: argparse.Namespace) -> dict:
    if args.generator != "itspm_latent_diffusion":
        raise ValueError("This entry point implements the observed-only latent generation head")
    start = time.time()
    if args.gpu != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != "cpu" else "cpu")

    data, mask, is_table1_dataset = load_table1_dataset(args)
    args.channels = data.shape[-1]
    data, mean, std, train_ids = normalize_train_test(data, args.train_frac, mask)
    split = len(train_ids)
    train_data = data[:split]
    test_data = data[split:]
    if len(test_data) < 2:
        raise ValueError("The evaluation split needs at least two sequences for discriminative scoring")
    # Materialize only observed training inputs. Evaluation sequences remain
    # outside the encoder and the generator's two training stages.
    obs_train = np.where(mask[:split] > 0, train_data, 0.0).astype(np.float32)
    times = np.broadcast_to(np.linspace(0.0, 1.0, args.seq_len, dtype=np.float32), data.shape[:2]).copy()

    table1_dataset = "sines" if args.dataset == "sine" else args.dataset
    missing_key = round(float(args.missing), 1)
    diffmn_ref = {
        metric: DIFF_MN_TABLE1[metric].get(missing_key, {}).get(table1_dataset)
        for metric in ("ds", "mdd", "kl")
    }
    test_eval = denormalize(test_data, mean, std)
    gauss = baseline_gaussian(train_data, len(test_data), args.seed + 99)
    gauss_eval = denormalize(gauss, mean, std)
    metrics = {
        "gaussian_baseline": table1_metrics(test_eval, gauss_eval, device, args),
    }
    out_dir = TASK_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ITSPMConfig(
        input_dim=args.channels,
        d_model=args.d_model,
        dropout=args.dropout,
        n_ref_points=args.n_ref_points,
        n_scales=args.n_scales,
        n_mixer_layers=args.n_mixer_layers,
        max_event_tokens=args.max_event_tokens,
        max_gap_tokens=args.max_gap_tokens,
    )

    ae = ITSPMLatentAutoencoder(cfg, args.seq_len, args.channels, args.latent_dim).to(device)
    opt_ae = torch.optim.AdamW(ae.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)

    x_train = torch.from_numpy(obs_train).to(device)
    m_train = torch.from_numpy(mask[:split]).to(device)
    t_train = torch.from_numpy(times[:split].copy()).to(device)

    for epoch in range(1, args.ae_epochs + 1):
        losses = []
        ae.train()
        for ids_np in iter_batches(split, args.batch_size, True, args.seed + epoch):
            ids = torch.as_tensor(ids_np, device=device)
            opt_ae.zero_grad(set_to_none=True)
            recon, _ = ae(t_train[ids], x_train[ids], m_train[ids])
            loss = observed_reconstruction_loss(recon, x_train[ids], m_train[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt_ae.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % max(1, args.log_every) == 0:
            print(f"ae_epoch={epoch:03d} loss={np.mean(losses):.6f}")

    freeze_autoencoder(ae)
    del opt_ae
    latents = []
    with torch.no_grad():
        for ids_np in iter_batches(split, args.batch_size, False, args.seed):
            ids = torch.as_tensor(ids_np, device=device)
            latents.append(ae.encode(t_train[ids], x_train[ids], m_train[ids]).detach().cpu())
    z_train = torch.cat(latents, dim=0).to(device)
    z_norm, z_mean, z_std = standardize_training_latents(z_train)

    ddpm = LatentDDPM(args.diffusion_steps, args.beta_start, args.beta_end, device)
    denoiser = LatentDenoiser(args.latent_dim, args.diffusion_hidden, args.diffusion_steps).to(device)
    opt_diff = torch.optim.AdamW(denoiser.parameters(), lr=args.diff_lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.diff_epochs + 1):
        losses = []
        denoiser.train()
        for ids_np in iter_batches(len(z_norm), args.batch_size, True, args.seed + 1000 + epoch):
            z0 = z_norm[torch.as_tensor(ids_np, device=device)]
            tt = torch.randint(0, args.diffusion_steps, (z0.size(0),), device=device)
            noise = torch.randn_like(z0)
            zt = ddpm.q_sample(z0, tt, noise)
            opt_diff.zero_grad(set_to_none=True)
            loss = F.mse_loss(denoiser(zt, tt), noise)
            loss.backward()
            opt_diff.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % max(1, args.log_every) == 0:
            print(f"diff_epoch={epoch:03d} loss={np.mean(losses):.6f}")

    fake = sample_sequences(ae, denoiser, ddpm, len(test_data), z_mean, z_std).cpu().numpy()
    fake_eval = denormalize(fake, mean, std)

    metrics.update({
        "itspm_latent_diffusion": table1_metrics(test_eval, fake_eval, device, args)
    })

    np.save(out_dir / "generated.npy", fake_eval.astype(np.float32))
    np.save(out_dir / "real_test.npy", test_eval.astype(np.float32))
    save_generation_checkpoint(out_dir / "model.pt", ae, denoiser, cfg, args, mean, std, z_mean, z_std)
    result = {
        "metrics": metrics,
        "diffmn_table1_reference": diffmn_ref,
        "table1_dataset_available": is_table1_dataset,
        "args": vars(args),
        "itspm_config": asdict(cfg),
        "elapsed_sec": time.time() - start,
        "source": (
            "One-for-all ITSPM backbone from parent models/ITSPM.py plus a "
            "latent generation task head. The ITSPM architecture is not modified."
        ),
        "paper_protocol": "observed_reconstruction_then_latent_diffusion",
        "training": {
            "reconstruction_scope": "observed_training_entries",
            "autoencoder_frozen_during_diffusion": True,
            "latent_statistics_scope": "training_latents",
        },
        "sampling": {
            "initial_state": "gaussian_noise",
            "evaluation_observations_used": False,
            "latent_inverse_standardization": True,
        },
        "postprocessing": {
            "marginal_calibration": False,
            "observed_value_clamping": False,
        },
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Irregular time series generation with ITSPM latent diffusion")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dataset", type=str, default="sines",
                   choices=["sine", "sines", "stocks", "energy", "mujoco", "polynomial"])
    p.add_argument("--generator", type=str, default="itspm_latent_diffusion",
                   choices=["itspm_latent_diffusion"])
    p.add_argument("--n_samples", type=int, default=1024)
    p.add_argument("--data_root", type=str, default=str(TASK_DIR / "table1_data"))
    p.add_argument("--seq_len", type=int, default=36)
    p.add_argument("--channels", type=int, default=5)
    p.add_argument("--missing", type=float, default=0.5)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--ae_epochs", type=int, default=40)
    p.add_argument("--diff_epochs", type=int, default=80)
    p.add_argument("--ds_iterations", type=int, default=2000)
    p.add_argument("--ds_batch_size", type=int, default=128)
    p.add_argument("--mdd_bins", type=int, default=20)
    p.add_argument("--ae_lr", type=float, default=1e-3)
    p.add_argument("--diff_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--diffusion_steps", type=int, default=50)
    p.add_argument("--diffusion_hidden", type=int, default=128)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n_ref_points", type=int, default=24)
    p.add_argument("--n_scales", type=int, default=2)
    p.add_argument("--n_mixer_layers", type=int, default=1)
    p.add_argument("--max_event_tokens", type=int, default=48)
    p.add_argument("--max_gap_tokens", type=int, default=24)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", type=str, default="outputs/default")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
