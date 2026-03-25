"""
transforms.py
-------------
Pure-numpy normalization applied to waveforms and 3D ACGs before VAE
training and inference.  These functions are the single source of truth —
use them both during training (in the Dataset) and at inference time when
encoding new sessions.

Normalization choices (agreed with user):
  WF  — peak-amplitude scale only (no flip, no baseline removal)
  ACG — transpose axes, per-sample max scale to [0,1], crop to positive lags
"""

from __future__ import annotations
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Waveform
# ══════════════════════════════════════════════════════════════════════════════

def normalize_waveforms(wf: np.ndarray) -> np.ndarray:
    """
    Baseline-subtract then peak-amplitude normalize per unit.

    Steps
    -----
    1. Subtract per-channel baseline (mean of the first 10 samples) so the
       pre-spike region is centred at zero on every channel.
    2. Divide all channels by the peak absolute value of the main channel
       (channel 0), so the main-channel peak lands at ±1.

    Parameters
    ----------
    wf : (N, n_channels, n_timepoints) float

    Returns
    -------
    (N, n_channels, n_timepoints) float32
    """
    wf       = np.asarray(wf, dtype=np.float32)
    baseline = wf[:, :, :10].mean(axis=2, keepdims=True)   # (N, C, 1)
    wf       = wf - baseline                                 # zero pre-spike region
    amp      = np.abs(wf[:, 0, :]).max(axis=1)              # (N,) peak of main ch
    amp      = np.where(amp == 0.0, 1.0, amp)               # guard silent units
    return wf / amp[:, np.newaxis, np.newaxis]


# ══════════════════════════════════════════════════════════════════════════════
#  3D ACG
# ══════════════════════════════════════════════════════════════════════════════

def normalize_acg3d(acg: np.ndarray) -> np.ndarray:
    """
    Prepare 3D ACG for VAE input.

    Steps
    -----
    1. Transpose (N, 201, 10) → (N, 10, 201)   [FR-bins × lag-bins]
    2. Per-sample max normalization → [0, 1]
    3. Crop to positive lags: [:, :, 100:] → (N, 10, 101)
    4. Add channel dim → (N, 1, 10, 101)

    Parameters
    ----------
    acg : (N, 201, 10) — full mirrored ACG in Hz, as saved by run_extraction

    Returns
    -------
    (N, 1, 10, 101) float32  — ready for ACGConvVAE input
    """
    acg = np.asarray(acg, dtype=np.float32)

    # 1. Transpose: (N, 201, 10) → (N, 10, 201)
    acg = acg.transpose(0, 2, 1)

    # 2. Per-sample max normalization → [0, 1]
    amax = acg.max(axis=(1, 2), keepdims=True)      # (N, 1, 1)
    amax = np.where(amax == 0.0, 1.0, amax)         # guard silent units
    acg  = np.nan_to_num(acg / amax, nan=0.0)       # silent units → 0

    # 3. Crop to positive lags only: index 100 is lag=0, 100: gives 101 bins
    acg = acg[:, :, 100:]                           # (N, 10, 101)

    # 4. Add channel dim for Conv2D
    acg = acg[:, np.newaxis, :, :]                  # (N, 1, 10, 101)

    return acg
