"""
data_store.py
-------------
SessionData -- loads a _features.npz file (written by run_extraction.py)
and provides indexed access to waveforms, ACGs and metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class SessionData:
    """
    Wraps a _features.npz file for the GUI.

    Attributes
    ----------
    n_units      : int
    unit_ids     : (n,) int64
    labels       : (n,) str  -- expert / 'unknown'
    mean_waveforms  : (n, 8, 81) float32
    std_waveforms   : (n, 8, 81) float32
    n_spikes_wf  : (n,) int64
    main_channels: (n,) int64
    ch_positions : (n, 8, 2) float64  -- µm coords of 8 extracted channels
    acg_1d       : (n, 201) float64  -- conditional rate [Hz], ±20 ms at 0.2 ms bins
    acg_3d       : (n, 201, 10) float64
    t_ms         : (201,) float64  -- lag axis for acg_1d
    t_log        : (201,) float64   -- log-lag axis for acg_3d
    fr_edges     : (n, 11) float64
    session_name : str
    table        : pd.DataFrame | None  -- loaded from companion _table.csv
    """

    def __init__(self, npz_path: str | Path):
        npz_path = Path(npz_path)
        self._npz_path = npz_path          # keep for save_labels_to_npz
        data = np.load(npz_path, allow_pickle=True)

        self.unit_ids      = data["unit_ids"]
        self.labels        = data["labels"].astype(object).copy()  # object dtype for arbitrary-length strings
        # Override with sidecar labels file if present (written by save_labels)
        _sidecar = npz_path.parent / (npz_path.stem + "_labels.npy")
        if _sidecar.exists():
            self.labels = np.load(_sidecar, allow_pickle=True).astype(object)
        self.mean_waveforms = data["mean_waveforms"]
        self.std_waveforms  = data["std_waveforms"]
        self.n_spikes_wf   = data["n_spikes_wf"]
        self.main_channels = data["main_channels"]
        self.ch_positions  = data["channel_positions_used"]
        self.acg_1d        = data["acg_1d"]
        self.acg_3d        = data["acg_3d"]
        self.t_ms          = data["t_ms"]
        self.t_log         = data["t_log"]
        self.fr_edges      = data["fr_edges"]
        self.session_name  = str(data["session_name"])
        self.rec_duration_s = float(data["rec_duration_s"])
        self.n_units       = len(self.unit_ids)

        # ── CCG pair data (may not exist in older .npz files) ────────────
        if "ccg_auto_labels" in data:
            self.ccg_auto_labels = data["ccg_auto_labels"].copy()
            self.ccg_pair_ids    = data["ccg_pair_ids"]       # (P, 2) int64
            self.ccg_counts      = data["ccg_counts"]         # (P, n_bins) float32
            self.ccg_t_ms        = data["ccg_t_ms"]           # (n_bins,) float64
            self.ccg_pair_dists  = data["ccg_pair_dists"]     # (P,) float32
            self.ccg_pair_types  = data["ccg_pair_types"]     # (P,) str
            self.ccg_pair_scores = data["ccg_pair_scores"]    # (P,) float32
            self.has_ccg         = True
            self.n_pairs         = len(self.ccg_pair_ids)
        else:
            self.ccg_auto_labels = np.array([""] * self.n_units)
            self.ccg_pair_ids    = np.zeros((0, 2), dtype=np.int64)
            self.ccg_counts      = np.zeros((0, 0), dtype=np.float32)
            self.ccg_t_ms        = np.zeros(0)
            self.ccg_pair_dists  = np.zeros(0, dtype=np.float32)
            self.ccg_pair_types  = np.array([], dtype=str)
            self.ccg_pair_scores = np.zeros(0, dtype=np.float32)
            self.has_ccg         = False
            self.n_pairs         = 0

        # ── MFB data (may not exist in older .npz files) ────────────
        if "mfb_tier" in data:
            self.mfb_tier  = data["mfb_tier"].copy()   # (n,) str
            self.mfb_score = data["mfb_score"].copy()  # (n,) float32
            self.has_mfb   = True
        else:
            self.mfb_tier  = np.array([""] * self.n_units)
            self.mfb_score = np.full(self.n_units, float("nan"), dtype=np.float32)
            self.has_mfb   = False

        # ── Classifier predictions (populated by GUI after running inference) ──
        self.clf_labels = np.array([""] * self.n_units, dtype=object)
        self.clf_conf   = np.full(self.n_units, float("nan"), dtype=np.float32)
        self.has_clf    = False

        # Build uid → array-index lookup
        self._uid_to_idx = {int(uid): i for i, uid in enumerate(self.unit_ids)}

        # Try to load companion _table.csv (same directory, same stem)
        csv_path = npz_path.parent / (npz_path.stem.replace("_features", "_table") + ".csv")
        if not csv_path.exists():
            # Also try replacing _features with nothing
            csv_path = npz_path.parent / (npz_path.stem + ".csv")
        self.table: pd.DataFrame | None = None
        if csv_path.exists():
            try:
                self.table = pd.read_csv(csv_path)
            except Exception:
                pass

    # ── Per-unit accessors ────────────────────────────────────────────────────

    def get_wf(self, i: int):
        """Return (mean_wf, std_wf, depth_um) for unit at index i."""
        mean = self.mean_waveforms[i]      # (8, 81)
        std  = self.std_waveforms[i]       # (8, 81)
        depth = float(self.ch_positions[i, 0, 1])
        return mean, std, depth

    def get_acg_1d(self, i: int):
        """Return (acg, t_ms) for unit at index i."""
        return self.acg_1d[i], self.t_ms

    def get_acg_3d(self, i: int):
        """Return (acg_3d, t_log) for unit at index i.
        acg_3d shape: (201, 10) -- (log-lag bins, FR-quantile bins)
        """
        return self.acg_3d[i], self.t_log

    def get_label(self, i: int) -> str:
        return str(self.labels[i])

    def get_depth(self, i: int) -> float:
        return float(self.ch_positions[i, 0, 1])

    def get_mean_fr(self, i: int) -> float:
        if self.table is not None and "mean_fr_hz" in self.table.columns:
            try:
                uid = int(self.unit_ids[i])
                rows = self.table[self.table["unit_id"] == uid]
                if not rows.empty:
                    return float(rows.iloc[0]["mean_fr_hz"])
            except Exception:
                pass
        return float("nan")

    def get_layer(self, i: int) -> str:
        if self.table is not None and "neuron_layer" in self.table.columns:
            try:
                uid = int(self.unit_ids[i])
                rows = self.table[self.table["unit_id"] == uid]
                if not rows.empty:
                    v = rows.iloc[0]["neuron_layer"]
                    return str(v) if pd.notna(v) else ""
            except Exception:
                pass
        return ""

    def get_c4_pred(self, i: int) -> str:
        if self.table is not None and "C4_predicted_cell_type" in self.table.columns:
            try:
                uid = int(self.unit_ids[i])
                rows = self.table[self.table["unit_id"] == uid]
                if not rows.empty:
                    v = rows.iloc[0]["C4_predicted_cell_type"]
                    return str(v) if pd.notna(v) else ""
            except Exception:
                pass
        return ""

    def get_mfb_tier(self, i: int) -> str:
        return str(self.mfb_tier[i]) if self.has_mfb else ""

    def get_mfb_score(self, i: int) -> float:
        return float(self.mfb_score[i]) if self.has_mfb else float("nan")

    def get_clf_label(self, i: int) -> str:
        return str(self.clf_labels[i]) if self.has_clf else ""

    def get_clf_conf(self, i: int) -> float:
        return float(self.clf_conf[i]) if self.has_clf else float("nan")

    def get_ccg_label(self, i: int) -> str:
        return str(self.ccg_auto_labels[i]) if i < len(self.ccg_auto_labels) else ""

    def uid_to_idx(self, uid: int) -> int:
        return self._uid_to_idx.get(uid, -1)

    def save_labels_to_npz(self):
        """Save labels to a small sidecar .npy file (fast — does not touch the npz)."""
        sidecar = self._npz_path.parent / (self._npz_path.stem + "_labels.npy")
        np.save(sidecar, self.labels)

    # ── Pair accessors ───────────────────────────────────────────────────────

    def get_pair_ccg(self, pair_idx: int, flip: bool = False) -> np.ndarray:
        """Return CCG counts for pair.  If *flip*, return CCG(B→A)."""
        ccg = self.ccg_counts[pair_idx].astype(np.float64)
        return ccg[::-1].copy() if flip else ccg

    def get_pair_uids(self, pair_idx: int) -> tuple[int, int]:
        """Return (uid_A, uid_B) for pair at *pair_idx*."""
        return int(self.ccg_pair_ids[pair_idx, 0]), int(self.ccg_pair_ids[pair_idx, 1])

    def get_pairs_for_unit(self, uid: int) -> np.ndarray:
        """Return pair indices involving *uid*, sorted by score descending."""
        if self.n_pairs == 0:
            return np.array([], dtype=np.int64)
        mask = (self.ccg_pair_ids[:, 0] == uid) | (self.ccg_pair_ids[:, 1] == uid)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return idxs
        order = np.argsort(self.ccg_pair_scores[idxs])[::-1]
        return idxs[order]

    def get_detected_pair_indices(self) -> np.ndarray:
        """Return indices of pairs where type != 'none', sorted by score."""
        if self.n_pairs == 0:
            return np.array([], dtype=np.int64)
        mask = self.ccg_pair_types != "none"
        idxs = np.where(mask)[0]
        order = np.argsort(self.ccg_pair_scores[idxs])[::-1]
        return idxs[order]

    def get_all_pair_indices_sorted(self) -> np.ndarray:
        """Return all pair indices, detected first (by score), then rest."""
        if self.n_pairs == 0:
            return np.array([], dtype=np.int64)
        det_mask = self.ccg_pair_types != "none"
        det = np.where(det_mask)[0]
        non = np.where(~det_mask)[0]
        det_sorted = det[np.argsort(self.ccg_pair_scores[det])[::-1]]
        non_sorted = non[np.argsort(self.ccg_pair_dists[non])]
        return np.concatenate([det_sorted, non_sorted])
