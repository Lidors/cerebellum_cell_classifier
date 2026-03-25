"""
datasets.py
-----------
PyTorch Dataset classes that load waveforms and 3D ACGs from one or many
NPZ feature files (as produced by run_extraction.py) and return normalised
float32 tensors ready for VAE training.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import normalize_waveforms, normalize_acg3d


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_npz_list(npz_paths: list[str | Path]) -> list[np.lib.npyio.NpzFile]:
    return [np.load(Path(p), allow_pickle=True) for p in npz_paths]


def _spikes_mask(npz, min_spikes: int) -> np.ndarray | None:
    """Return boolean mask for units with enough spikes, or None (keep all)."""
    if min_spikes <= 0:
        return None
    if "n_spikes_wf" in npz:
        return npz["n_spikes_wf"] >= min_spikes
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Waveform dataset
# ══════════════════════════════════════════════════════════════════════════════

class WFDataset(Dataset):
    """
    Multi-NPZ waveform dataset for unsupervised VAE training.

    Loads ``mean_waveforms`` from all provided NPZ files, concatenates them,
    applies peak-amplitude normalization, and returns
    ``(1, n_chan_use, n_timepoints)`` float32 tensors.

    Parameters
    ----------
    npz_paths  : list of paths to .npz feature files
    min_spikes : minimum value of ``n_spikes_wf`` to include a unit;
                 0 = include all (default)
    n_chan_use : how many channels to keep (default None = all).
                 Channels are ordered by proximity to the main channel
                 (index 0 = main), so n_chan_use=4 keeps the 4 closest.
    """

    def __init__(
        self,
        npz_paths:  list[str | Path],
        min_spikes: int = 0,
        n_chan_use: "int | None" = None,
    ):
        wf_list = []
        for path in npz_paths:
            npz  = np.load(Path(path), allow_pickle=True)
            wf   = npz["mean_waveforms"].astype(np.float32)   # (N, C, T)
            mask = _spikes_mask(npz, min_spikes)
            if mask is not None:
                wf = wf[mask]
            wf_list.append(wf)

        wf_all = np.concatenate(wf_list, axis=0)              # (N_total, C, T)
        wf_all = normalize_waveforms(wf_all)                   # peak scale

        # Optionally keep only the first n_chan_use channels
        if n_chan_use is not None:
            wf_all = wf_all[:, :n_chan_use, :]                 # (N, n_chan_use, T)

        # Add channel dim → (N, 1, C, T) for Conv2D
        wf_all = wf_all[:, np.newaxis, :, :]
        self._data = torch.from_numpy(wf_all)

        print(f"WFDataset: {len(self._data)} units from {len(npz_paths)} "
              f"file(s).  Sample shape: {tuple(self._data.shape[1:])}")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  3D ACG dataset
# ══════════════════════════════════════════════════════════════════════════════

class ACGDataset(Dataset):
    """
    Multi-NPZ 3D-ACG dataset for unsupervised VAE training.

    Loads ``acg_3d`` from all provided NPZ files, concatenates them, applies
    normalization (transpose → per-sample max → crop → add channel dim),
    and returns ``(1, 10, 101)`` float32 tensors.

    Parameters
    ----------
    npz_paths  : list of paths to .npz feature files
    min_spikes : minimum value of ``n_spikes_wf`` to include a unit;
                 0 = include all (default)
    """

    def __init__(self, npz_paths: list[str | Path], min_spikes: int = 0):
        acg_list = []
        for path in npz_paths:
            npz  = np.load(Path(path), allow_pickle=True)
            acg  = npz["acg_3d"].astype(np.float32)           # (N, 201, 10)
            mask = _spikes_mask(npz, min_spikes)
            if mask is not None:
                acg = acg[mask]
            acg_list.append(acg)

        acg_all = np.concatenate(acg_list, axis=0)            # (N_total, 201, 10)
        acg_all = normalize_acg3d(acg_all)                     # (N_total, 1, 10, 101)
        self._data = torch.from_numpy(acg_all)

        print(f"ACGDataset: {len(self._data)} units from {len(npz_paths)} "
              f"file(s).  Sample shape: {tuple(self._data.shape[1:])}")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]
