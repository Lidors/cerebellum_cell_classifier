"""
run_extraction.py
-----------------
Extract mean waveforms and ACGs for all good units in a Kilosort 4 session
and save a consolidated feature file + per-unit table.

Usage (Python)
--------------
    from cerebellum_cell_classifier.run_extraction import run_extraction

    run_extraction(
        session_path = r"E:\\data\\AA23\\AA23_05",
        bin_path     = r"E:\\data\\AA23\\AA23_05\\AA23_05_g0_tcat.imec0.ap.bin",
        labels       = {5: "PC", 124: "MLI", 340: "MF"},   # optional
        output_path  = r"C:\\data\\features",
    )

Usage (command line)
--------------------
    python run_extraction.py \\
        --session  E:\\data\\AA23\\AA23_05 \\
        --bin      E:\\data\\AA23\\AA23_05\\AA23_05_g0_tcat.imec0.ap.bin \\
        --labels   5:PC 124:MLI 340:MF \\
        --output   C:\\data\\features

Output files (in output_path/)
-------------------------------
    {session_name}_features.npz   — all arrays (waveforms + ACGs + axes + metadata)
    {session_name}_table.csv      -- per-unit table (depth, FR, label, ...)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project imports ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cerebellum_cell_classifier.io.kilosort import load_kilosort
from cerebellum_cell_classifier.features.waveform import build_waveform_features
from cerebellum_cell_classifier.features.acg import build_acg_features
from cerebellum_cell_classifier.features.ccg import build_ccg_labels


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def run_extraction(
    session_path: str | Path,
    bin_path: str | Path | None = None,
    labels: dict[int, str] | None = None,
    output_path: str | Path | None = None,
    unit_ids: "list[int] | np.ndarray | None" = None,
    n_channels_total: int = 385,
    sample_rate: float = 30_000.0,
    max_spikes: int = 3_000,
    run_ccg_labeling: bool = True,
    ccg_max_distance_um: float = 200.0,
    ccg_min_pc_fr_hz: float = 30.0,
    verbose: bool = True,
) -> dict:
    """
    Run the full feature extraction pipeline for one session.

    Parameters
    ----------
    session_path : str or Path
        Kilosort 4 output directory (contains *.npy + cluster_info.tsv).
    bin_path : str or Path, optional
        Path to the raw ``*ap.bin`` binary.  Auto-detected when omitted.
    labels : dict {int -> str}, optional
        Expert cell-type labels, e.g. ``{5: 'PC', 124: 'MLI'}``.
        Unlabelled units receive ``'unknown'``.
    output_path : str or Path, optional
        Directory where output files are written.  Defaults to
        ``session_path/features/``.
    unit_ids : array-like of int, optional
        Cluster IDs to process.  Defaults to all good units from
        cluster_info.tsv.
    n_channels_total : int
        Total channels in the binary file (default 385 for NP1/NP2).
    sample_rate : float
        Sampling rate in Hz (default 30 000).
    max_spikes : int
        Max spikes per unit for waveform extraction (default 3 000).
    run_ccg_labeling : bool
        If True (default), auto-label PC/CF and MLI units using pairwise
        CCG analysis.  Expert labels supplied via ``labels`` are preserved.
    ccg_max_distance_um : float
        Maximum inter-channel distance (µm) for CCG pair candidates
        (default 200).
    ccg_min_pc_fr_hz : float
        Minimum mean firing rate (Hz) required for a PC candidate
        (default 30).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Contains all extracted arrays (same data as the saved .npz).
        Keys: unit_ids, labels, mean_waveforms, std_waveforms, acg_1d,
              acg_3d, t_ms, t_log, main_channels, used_channels,
              channel_positions_used, n_spikes_wf, fr_edges, table,
              pc_cf_pairs, mli_units.
    """
    t0 = time.perf_counter()
    session_path = Path(session_path)
    session_name = session_path.name

    if output_path is None:
        output_path = session_path / "features"
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Kilosort data ─────────────────────────────────────────────────
    n_steps = 5 if run_ccg_labeling else 4

    if verbose:
        print("=" * 60)
        print(f"Session : {session_name}")
        print(f"KS dir  : {session_path}")
        print("=" * 60)
        print(f"\n[1/{n_steps}] Loading Kilosort data ...")

    ks = load_kilosort(session_path, sample_rate=sample_rate)

    # Determine units to process
    if unit_ids is None:
        unit_ids_arr = ks.good_units
        if verbose:
            print(f"  {len(unit_ids_arr)} good units from cluster_info.tsv")
    else:
        unit_ids_arr = np.asarray(unit_ids, dtype=np.int64)
        if verbose:
            print(f"  {len(unit_ids_arr)} units provided by caller")

    # Recording duration (for FR computation)
    rec_duration_s = ks.spike_times[-1] / sample_rate if len(ks.spike_times) else 0.0

    # ── 2. Waveform features ─────────────────────────────────────────────────
    if verbose:
        print(f"\n[2/{n_steps}] Extracting mean waveforms ...")

    wf_result = build_waveform_features(
        session_path     = session_path,
        ks_data          = ks,
        unit_ids         = unit_ids_arr,
        cell_type_labels = labels,
        bin_path         = bin_path,
        n_channels_total = n_channels_total,
        max_spikes       = max_spikes,
        verbose          = verbose,
    )

    # ── 3. ACG features ───────────────────────────────────────────────────────
    if verbose:
        print(f"\n[3/{n_steps}] Computing ACGs ...")

    acg_result = build_acg_features(
        unit_ids       = unit_ids_arr,
        spike_times    = ks.spike_times,
        spike_clusters = ks.spike_clusters,
        sample_rate    = sample_rate,
        lag_ms         = 2000.0,
        bin_ms         = 1.0,
        n_fr_bins      = 10,
        n_log_bins     = 100,
        log_start_ms   = 0.8,
        verbose        = verbose,
    )

    # ── 4. CCG-based auto-labeling ────────────────────────────────────────────
    pc_cf_pairs: list = []
    mli_units:   list = []

    if run_ccg_labeling:
        if verbose:
            print(f"\n[4/{n_steps}] CCG-based cell-type labeling ...")

        # Main-channel position per unit: shape (N, 2)
        unit_positions_um = wf_result["channel_positions_used"][:, 0, :]

        ccg_result = build_ccg_labels(
            unit_ids          = unit_ids_arr,
            spike_times       = ks.spike_times,
            spike_clusters    = ks.spike_clusters,
            unit_positions_um = unit_positions_um,
            initial_labels    = wf_result["cell_type_labels"],
            sample_rate       = sample_rate,
            max_distance_um   = ccg_max_distance_um,
            min_pc_fr_hz      = ccg_min_pc_fr_hz,
            verbose           = verbose,
        )

        # Update labels with CCG-derived labels
        wf_result["cell_type_labels"] = ccg_result["labels"]
        pc_cf_pairs = ccg_result["pc_cf_pairs"]
        mli_units   = ccg_result["mli_units"]
        ccg_auto_labels = ccg_result["ccg_auto_labels"]

    # ── 5. Build per-unit table ───────────────────────────────────────────────
    step_tbl = 5 if run_ccg_labeling else 4
    if verbose:
        print(f"\n[{step_tbl}/{n_steps}] Building unit table ...")

    table = _build_unit_table(
        unit_ids         = unit_ids_arr,
        ks               = ks,
        session_path     = session_path,
        labels           = wf_result["cell_type_labels"],
        main_channels    = wf_result["main_channels"],
        ch_positions     = wf_result["channel_positions_used"],
        n_spikes_wf      = wf_result["n_spikes"],
        sample_rate      = sample_rate,
        rec_duration_s   = rec_duration_s,
        session_name     = session_name,
    )

    # ── 6. Save ───────────────────────────────────────────────────────────────
    npz_path = output_path / f"{session_name}_features.npz"
    csv_path = output_path / f"{session_name}_table.csv"

    save_dict = dict(
        # identity
        unit_ids              = wf_result["unit_ids"],
        labels                = wf_result["cell_type_labels"],
        # waveforms
        mean_waveforms        = wf_result["mean_waveforms"],
        std_waveforms         = wf_result["std_waveforms"],
        n_spikes_wf           = wf_result["n_spikes"],
        main_channels         = wf_result["main_channels"],
        used_channels         = wf_result["used_channels"],
        channel_positions_used= wf_result["channel_positions_used"],
        # ACGs
        acg_1d                = acg_result["acg_1d"],
        acg_3d                = acg_result["acg_3d"],
        t_ms                  = acg_result["t_ms"],
        t_log                 = acg_result["t_log"],
        fr_edges              = acg_result["fr_edges"],
        # session metadata (stored as 0-d object arrays)
        session_name          = np.array(session_name),
        session_path          = np.array(str(session_path)),
        sample_rate           = np.array(sample_rate),
        rec_duration_s        = np.array(rec_duration_s),
    )

    # Add CCG data if available
    if run_ccg_labeling:
        save_dict["ccg_auto_labels"] = ccg_auto_labels
        save_dict["ccg_pair_ids"]    = ccg_result["pair_unit_ids"]
        save_dict["ccg_counts"]      = ccg_result["pair_ccgs"]
        save_dict["ccg_t_ms"]        = ccg_result["ccg_t_ms"]
        save_dict["ccg_pair_dists"]  = ccg_result["pair_dists"]
        save_dict["ccg_pair_types"]  = ccg_result["pair_types"]
        save_dict["ccg_pair_scores"] = ccg_result["pair_scores"]

    np.savez_compressed(npz_path, **save_dict)

    table.to_csv(csv_path, index=False)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\n{'-'*60}")
        print(f"Done in {elapsed:.1f} s")
        print(f"  features : {npz_path}")
        print(f"  table    : {csv_path}")
        print(f"{'-'*60}\n")

    return {
        "unit_ids":               wf_result["unit_ids"],
        "labels":                 wf_result["cell_type_labels"],
        "mean_waveforms":         wf_result["mean_waveforms"],
        "std_waveforms":          wf_result["std_waveforms"],
        "n_spikes_wf":            wf_result["n_spikes"],
        "main_channels":          wf_result["main_channels"],
        "used_channels":          wf_result["used_channels"],
        "channel_positions_used": wf_result["channel_positions_used"],
        "acg_1d":                 acg_result["acg_1d"],
        "acg_3d":                 acg_result["acg_3d"],
        "t_ms":                   acg_result["t_ms"],
        "t_log":                  acg_result["t_log"],
        "fr_edges":               acg_result["fr_edges"],
        "pc_cf_pairs":            pc_cf_pairs,
        "mli_units":              mli_units,
        "table":                  table,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_unit_table(
    unit_ids: np.ndarray,
    ks,
    session_path: Path,
    labels: np.ndarray,
    main_channels: np.ndarray,
    ch_positions: np.ndarray,      # (n_units, 8, 2) µm
    n_spikes_wf: np.ndarray,
    sample_rate: float,
    rec_duration_s: float,
    session_name: str,
) -> pd.DataFrame:
    """
    Build a per-unit DataFrame with depth, FR, label, and session columns.

    Columns always present:
        session, unit_id, label, ks_label, main_channel, depth_um, x_um,
        n_spikes_total, n_spikes_wf, mean_fr_hz

    Columns included when present in cluster_info.tsv:
        ContamPct, SNR, Amplitude, waveform_width, neuron_layer,
        C4_predicted_cell_type
    """
    # Load cluster_info.tsv to merge extra columns
    ci_path = session_path / "cluster_info.tsv"
    ci = pd.read_csv(ci_path, sep="\t")
    # Normalise cluster ID column name
    if "cluster_id" not in ci.columns and "id" in ci.columns:
        ci = ci.rename(columns={"id": "cluster_id"})
    ci = ci.set_index("cluster_id")

    # Optional extra columns to pull from cluster_info if available
    _OPTIONAL_COLS = [
        "group",                  # KS label: good / mua / noise
        "ContamPct",
        "SNR",
        "Amplitude",
        "waveform_width",
        "neuron_layer",
        "C4_predicted_cell_type",
    ]
    extra_cols = [c for c in _OPTIONAL_COLS if c in ci.columns]

    rows = []
    for i, uid in enumerate(unit_ids):
        uid_int = int(uid)

        # Main channel and depth (y-position of main channel in µm)
        main_ch  = int(main_channels[i])
        depth_um = float(ch_positions[i, 0, 1]) if main_ch >= 0 else float("nan")
        x_um     = float(ch_positions[i, 0, 0]) if main_ch >= 0 else float("nan")

        # Spike count and mean FR from raw spike data
        st = ks.spike_times[ks.spike_clusters == uid]
        n_spikes_total = len(st)
        mean_fr = n_spikes_total / rec_duration_s if rec_duration_s > 0 else 0.0

        row = {
            "session":        session_name,
            "unit_id":        uid_int,
            "label":          labels[i],
            "main_channel":   main_ch,
            "depth_um":       round(depth_um, 1),
            "x_um":           round(x_um, 1),
            "n_spikes_total": n_spikes_total,
            "n_spikes_wf":    int(n_spikes_wf[i]),
            "mean_fr_hz":     round(mean_fr, 2),
        }

        # Merge columns from cluster_info.tsv
        if uid_int in ci.index:
            ci_row = ci.loc[uid_int]
            for col in extra_cols:
                row[col] = ci_row[col]
        else:
            for col in extra_cols:
                row[col] = None

        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract waveform + ACG features from a KS4 session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--session",  required=True,
                   help="Path to Kilosort 4 output directory.")
    p.add_argument("--bin",      default=None,
                   help="Path to *.ap.bin file (auto-detected if omitted).")
    p.add_argument("--output",   default=None,
                   help="Output directory (default: <session>/features/).")
    p.add_argument("--labels",   nargs="*", default=None,
                   metavar="ID:LABEL",
                   help="Expert labels as space-separated id:label pairs, "
                        "e.g. --labels 5:PC 124:MLI")
    p.add_argument("--units",    nargs="*", default=None, type=int,
                   metavar="ID",
                   help="Cluster IDs to process (default: all good units).")
    p.add_argument("--n_ch",     type=int, default=385,
                   help="Total channels in binary (default: 385).")
    p.add_argument("--sr",       type=float, default=30_000.0,
                   help="Sample rate in Hz (default: 30000).")
    p.add_argument("--max_spikes", type=int, default=3_000,
                   help="Max spikes per unit for WF (default: 3000).")
    p.add_argument("--no_ccg", action="store_true",
                   help="Disable CCG-based auto-labeling (PC/CF/MLI).")
    p.add_argument("--ccg_dist", type=float, default=200.0,
                   help="Max inter-channel distance for CCG pairs in µm (default: 200).")
    p.add_argument("--ccg_min_fr", type=float, default=30.0,
                   help="Min mean FR (Hz) for PC candidates in CCG (default: 30).")
    return p.parse_args()


def _parse_labels(label_strs: "list[str] | None") -> "dict[int, str] | None":
    if not label_strs:
        return None
    out = {}
    for token in label_strs:
        if ":" not in token:
            raise ValueError(f"Label token '{token}' must be in 'id:label' format.")
        uid, lbl = token.split(":", 1)
        out[int(uid)] = lbl
    return out


if __name__ == "__main__":
    args = _parse_args()
    run_extraction(
        session_path         = args.session,
        bin_path             = args.bin,
        labels               = _parse_labels(args.labels),
        output_path          = args.output,
        unit_ids             = args.units,
        n_channels_total     = args.n_ch,
        sample_rate          = args.sr,
        max_spikes           = args.max_spikes,
        run_ccg_labeling     = not args.no_ccg,
        ccg_max_distance_um  = args.ccg_dist,
        ccg_min_pc_fr_hz     = args.ccg_min_fr,
    )
