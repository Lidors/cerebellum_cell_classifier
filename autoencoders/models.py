"""
models.py
---------
Beta-VAE architectures for waveform (WF) and 3D ACG dimensionality reduction.

Adapted from Cell 2025 (Beau, Herzfeld, Naveros et al.):
"A deep learning strategy to identify cell types across species from
high-density extracellular recordings", Cell 188(8):2218-2234.
GitHub: m-beau/NeuroPyxels, npyx/c4/

Adaptations for this project
-----------------------------
WF  : 8 channels × 81 timepoints (vs 4 × 90 in paper).  Encoder output
      size is computed dynamically so the architecture works for any input
      shape without manual adjustment.
ACG : identical to paper — input (1, 10, 101), latent 10-dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  Waveform Conv-VAE
#  Input : (batch, 1, n_channels, n_timepoints)  e.g. (B, 1, 8, 81)
#  Latent: 10-dim
# ══════════════════════════════════════════════════════════════════════════════

class _WFEncoder(nn.Module):
    def __init__(self, n_channels: int, n_timepoints: int, latent_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            # Convolve along the time axis
            nn.Conv2d(1, 4, kernel_size=(1, 8)),
            nn.AvgPool2d((1, 2)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # Convolve along the channel axis
            nn.Conv2d(4, 8, kernel_size=(3, 1)),
            nn.AvgPool2d((1, 2)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        # Compute flat size for any (n_channels, n_timepoints) input
        with torch.no_grad():
            dummy     = torch.zeros(1, 1, n_channels, n_timepoints)
            flat_size = self.conv(dummy).flatten(1).shape[1]

        self.fc     = nn.Linear(flat_size, 20)
        self.mu     = nn.Linear(20, latent_dim)
        self.logvar = nn.Linear(20, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.conv(x).flatten(1)
        h = self.fc(h)          # linear — no ReLU before mu/logvar heads
        return self.mu(h), self.logvar(h)


class _WFDecoder(nn.Module):
    def __init__(self, n_channels: int, n_timepoints: int, latent_dim: int):
        super().__init__()
        out_size = n_channels * n_timepoints          # 8 × 81 = 648
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, out_size),
            # No output activation — MSE loss drives outputs toward [-1, 1].
            # Tanh was removed because it asymptotes and can never reach ±1,
            # causing vanishing gradients exactly where the spike peak lives.
        )
        self._nc = n_channels
        self._nt = n_timepoints

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, self._nc, self._nt)


class WFConvVAE(nn.Module):
    """
    Convolutional Beta-VAE for multi-channel spike waveforms.

    Parameters
    ----------
    n_channels   : number of recorded channels (default 8)
    n_timepoints : samples per channel (default 81)
    latent_dim   : bottleneck dimensionality (default 10)
    beta         : KL weight — constant during training (default 5)
    """

    def __init__(
        self,
        n_channels:   int   = 8,
        n_timepoints: int   = 81,
        latent_dim:   int   = 10,
        beta:         float = 5.0,
    ):
        super().__init__()
        self.encoder = _WFEncoder(n_channels, n_timepoints, latent_dim)
        self.decoder = _WFDecoder(n_channels, n_timepoints, latent_dim)
        self.beta    = beta

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu   # deterministic at inference

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent mean (no sampling) — use for downstream features."""
        self.eval()
        mu, _ = self.encoder(x)
        return mu


# ══════════════════════════════════════════════════════════════════════════════
#  3D ACG Conv-VAE
#  Input : (batch, 1, 10, 101)   [1 channel × FR-bins × lag-bins]
#  Latent: 10-dim
# ══════════════════════════════════════════════════════════════════════════════

class _ACGEncoder(nn.Module):
    def __init__(self, latent_dim: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            # Convolve along the lag axis (width=10)
            nn.Conv2d(1,  8, kernel_size=(1, 10)),
            nn.AvgPool2d((2, 2)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # Convolve along the FR axis (height=5)
            nn.Conv2d(8, 16, kernel_size=(5, 1)),
            nn.AvgPool2d((1, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # For input (1, 10, 101) flat size = 16 × 1 × 23 = 368
        with torch.no_grad():
            dummy     = torch.zeros(1, 1, 10, 101)
            flat_size = self.conv(dummy).flatten(1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu     = nn.Linear(200, latent_dim)
        self.logvar = nn.Linear(200, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.conv(x).flatten(1)
        h = self.fc(h)
        return self.mu(h), self.logvar(h)


class _ACGDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        # 1 × 10 × 101 = 1010
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 1010),
            nn.Sigmoid(),    # ACG values in [0, 1] after normalisation
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 10, 101)


class ACGConvVAE(nn.Module):
    """
    Convolutional Beta-VAE for 3D ACG (FR-bins × lag-bins).

    Parameters
    ----------
    latent_dim : bottleneck dimensionality (default 10)
    beta       : maximum KL weight; use with beta annealing in train.py
                 (default 5)
    dropout    : encoder dropout rate (default 0.2)
    """

    def __init__(
        self,
        latent_dim: int   = 10,
        beta:       float = 5.0,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.encoder = _ACGEncoder(latent_dim, dropout)
        self.decoder = _ACGDecoder(latent_dim)
        self.beta    = beta

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent mean (no sampling) — use for downstream features."""
        self.eval()
        mu, _ = self.encoder(x)
        return mu


# ══════════════════════════════════════════════════════════════════════════════
#  ELBO loss
# ══════════════════════════════════════════════════════════════════════════════

def elbo_loss(
    recon:            torch.Tensor,
    target:           torch.Tensor,
    mu:               torch.Tensor,
    logvar:           torch.Tensor,
    beta:             float = 5.0,
    short_lag_weight: float = 1.0,
    n_short:          int   = 20,
    amplitude_weight: float = 0.0,
) -> tuple[torch.Tensor, float, float]:
    """
    Evidence Lower BOund loss: reconstruction (MSE) + beta × KL.

    Parameters
    ----------
    recon, target     : model output and input batch
    mu, logvar        : encoder output
    beta              : KL weight (use annealed value during ACG training)
    short_lag_weight  : extra MSE weight for the first ``n_short`` lag bins
                        (ACG only — set 1.5 for ACG, 1.0 for WF)
    n_short           : number of short-lag bins to up-weight (default 20)
    amplitude_weight  : if > 0, the per-sample MSE weight is scaled by
                        (1 + amplitude_weight * |target|), so high-amplitude
                        samples (spike peaks) contribute more to the loss.
                        Useful for WF where flat baseline samples dominate MSE.
                        Typical value: 3.0–5.0.  0 = disabled (default).

    Returns
    -------
    (total_loss, recon_loss_scalar, kl_loss_scalar)
    """
    if short_lag_weight != 1.0:
        weight = torch.ones_like(target)
        weight[..., :n_short] = short_lag_weight
        recon_loss = (weight * (recon - target) ** 2).mean()
    elif amplitude_weight > 0.0:
        weight     = 1.0 + amplitude_weight * target.abs()
        recon_loss = (weight * (recon - target) ** 2).mean()
    else:
        recon_loss = F.mse_loss(recon, target)

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total   = recon_loss + beta * kl_loss

    return total, float(recon_loss.item()), float(kl_loss.item())
