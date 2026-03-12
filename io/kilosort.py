"""
kilosort.py
-----------
Load Kilosort 4 output files for a Neuropixels session.

Required files in the session directory:
    spike_times.npy       - spike times in samples (int64)
    spike_clusters.npy    - cluster ID for each spike (int64)
    channel_map.npy       - index → hardware channel number, 0-based (int64)
    channel_positions.npy - x/y position of each channel (float64, µm)
    cluster_info.tsv      - cluster labels; must have 'ch', 'group'/'KSLabel' columns

Optional:
    spike_positions.npy   - estimated x/y position of each spike (only needed
                            if use_template_channel=False in waveform extraction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Data container ──────────────────────────────────────────────────────────────

@dataclass
class KilosortData:
    """
    Container for all Kilosort output arrays needed for feature extraction.

    Attributes
    ----------
    spike_times : (n_spikes,) int64
        Spike times in samples.
    spike_clusters : (n_spikes,) int64
        Cluster ID assigned to each spike.
    channel_map : (n_active_channels,) int64
        Maps compressed index → hardware channel number (0-based).
        channel_positions[i] corresponds to channel_map[i].
    channel_positions : (n_active_channels, 2) float64
        Physical x/y position (µm) of each active channel on the probe.
        column 0 = x (shank), column 1 = y (depth, 0 = tip).
    cluster_channels : dict[int, int]
        Maps cluster_id → main hardware channel (0-based), taken from the
        'ch' column of cluster_info.tsv.  This is the template-assigned
        channel — the safest and most reliable choice for KS4.
    good_units : (n_good,) int64
        Cluster IDs labeled 'good' in cluster_info.tsv.
    sample_rate : float
        Recording sample rate in Hz (default 30 000).
    spike_positions : (n_spikes, 2) float64 or None
        Estimated x/y position of each spike.  Only loaded when present
        (KS4 writes spike_positions.npy; older versions may not).
    """

    spike_times:       np.ndarray
    spike_clusters:    np.ndarray
    channel_map:       np.ndarray
    channel_positions: np.ndarray
    cluster_channels:  dict              # {cluster_id (int) → hw_channel (int)}
    good_units:        np.ndarray
    sample_rate:       float = 30_000.0
    spike_positions:   Optional[np.ndarray] = field(default=None, repr=False)
    # Kilosort template arrays (optional — used for MFB/NAW waveform features)
    templates:         Optional[np.ndarray] = field(default=None, repr=False)
    # (n_templates, n_samples, n_channels) float32
    spike_templates:   Optional[np.ndarray] = field(default=None, repr=False)
    # (n_spikes,) int64 — 0-based template index per spike
    whitening_mat_inv: Optional[np.ndarray] = field(default=None, repr=False)
    # (n_channels, n_channels) float64 — inverse whitening matrix


# ── Public API ──────────────────────────────────────────────────────────────────

def load_kilosort(
    session_path: str | Path,
    sample_rate: float = 30_000.0,
) -> KilosortData:
    """
    Load Kilosort 4 output files from a session directory.

    Parameters
    ----------
    session_path : str or Path
        Directory containing Kilosort output (.npy files + cluster_info.tsv).
    sample_rate : float
        Recording sample rate in Hz (default: 30 000).

    Returns
    -------
    KilosortData

    Raises
    ------
    FileNotFoundError
        If a required file is missing.
    ValueError
        If cluster_info.tsv has no recognisable label or channel column.
    """
    path = Path(session_path)
    _check_required_files(path)

    spike_times       = np.load(path / "spike_times.npy").flatten().astype(np.int64)
    spike_clusters    = np.load(path / "spike_clusters.npy").flatten().astype(np.int64)
    channel_map       = np.load(path / "channel_map.npy").flatten().astype(np.int64)
    channel_positions = np.load(path / "channel_positions.npy").astype(np.float64)

    # spike_positions is optional (KS4 writes it, older versions may not)
    sp_pos_path = path / "spike_positions.npy"
    spike_positions = (
        np.load(sp_pos_path).astype(np.float64) if sp_pos_path.exists() else None
    )

    cluster_info    = _load_cluster_info(path / "cluster_info.tsv")
    good_units      = _extract_good_units(cluster_info)
    cluster_channels = _extract_cluster_channels(cluster_info)

    # Optional template arrays (for MFB/NAW waveform feature extraction)
    tpl_path = path / "templates.npy"
    templates = np.load(tpl_path).astype(np.float32) if tpl_path.exists() else None

    st_path = path / "spike_templates.npy"
    spike_templates = (
        np.load(st_path).flatten().astype(np.int64) if st_path.exists() else None
    )

    winv_path = path / "whitening_mat_inv.npy"
    whitening_mat_inv = (
        np.load(winv_path).astype(np.float64) if winv_path.exists() else None
    )

    return KilosortData(
        spike_times=spike_times,
        spike_clusters=spike_clusters,
        channel_map=channel_map,
        channel_positions=channel_positions,
        cluster_channels=cluster_channels,
        good_units=good_units,
        sample_rate=sample_rate,
        spike_positions=spike_positions,
        templates=templates,
        spike_templates=spike_templates,
        whitening_mat_inv=whitening_mat_inv,
    )


# ── Internal helpers ─────────────────────────────────────────────────────────────

# spike_positions.npy removed from required list — it is optional for KS4
_REQUIRED_FILES = [
    "spike_times.npy",
    "spike_clusters.npy",
    "channel_map.npy",
    "channel_positions.npy",
    "cluster_info.tsv",
]


def _check_required_files(path: Path) -> None:
    missing = [f for f in _REQUIRED_FILES if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Kilosort files in {path}:\n  " + "\n  ".join(missing)
        )


def _load_cluster_info(tsv_path: Path) -> pd.DataFrame:
    """Read cluster_info.tsv into a DataFrame with a validated 'cluster_id' column."""
    df = pd.read_csv(tsv_path, sep="\t")

    # Normalise cluster-ID column name
    if "cluster_id" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "cluster_id"})
        else:
            raise ValueError(
                f"cluster_info.tsv has no 'cluster_id' or 'id' column. "
                f"Columns: {df.columns.tolist()}"
            )
    return df


def _extract_good_units(df: pd.DataFrame) -> np.ndarray:
    """Return cluster IDs where group == 'good' (KS4) or KSLabel == 'good'."""
    if "group" in df.columns:
        mask = df["group"] == "good"
    elif "KSLabel" in df.columns:
        mask = df["KSLabel"] == "good"
    else:
        raise ValueError(
            f"cluster_info.tsv has no 'group' or 'KSLabel' column. "
            f"Columns: {df.columns.tolist()}"
        )
    return df.loc[mask, "cluster_id"].values.astype(np.int64)


def _extract_cluster_channels(df: pd.DataFrame) -> dict:
    """
    Return {cluster_id → main_hw_channel} from the 'ch' column.

    The 'ch' column in KS4's cluster_info.tsv holds the hardware channel
    index (0-based) of the template's peak amplitude — i.e. the main channel.
    """
    if "ch" not in df.columns:
        raise ValueError(
            "cluster_info.tsv has no 'ch' column (main channel per cluster). "
            f"Columns found: {df.columns.tolist()}\n"
            "This column is written by Kilosort 4.  For older Kilosort versions "
            "use use_template_channel=False and provide spike_positions.npy."
        )
    return {
        int(row.cluster_id): int(row.ch)
        for row in df[["cluster_id", "ch"]].itertuples(index=False)
    }
