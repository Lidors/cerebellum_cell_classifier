"""
waveform.py
-----------
Extract mean (and std) spike waveforms from a raw Neuropixels binary file.

Python port of:
    process_spike_waveforms_np.m  (main pipeline)
    dat2spk.m                     (raw waveform reading)
    realign_spikes.m              (peak-alignment)

Main public function
--------------------
build_waveform_features(session_path, ks_data, unit_ids, cell_type_labels, ...)
    → dict with mean_waveforms, std_waveforms, main_channels, used_channels

Channel selection strategy
---------------------------
use_template_channel=True  (default, recommended for KS4)
    The main channel comes directly from the 'ch' column of cluster_info.tsv —
    the channel where the cluster's best-matching template has peak amplitude.
    The 8 channels extracted are the n_channels_extract physically nearest to
    that channel (by µm distance from channel_positions.npy), main channel first.

use_template_channel=False  (requires spike_positions.npy)
    For each spike, find the channel closest to the spike's estimated position;
    the most frequently chosen channel across spikes = main channel.
    Useful for Kilosort versions that do not write a 'ch' column.

Binary file layout
-------------------
Neuropixels .ap.bin files are stored as interleaved int16 samples:
    row = time sample,  column = hardware channel
    shape → (n_total_samples, n_channels_total)
NP2.0 default: 385 channels (384 neural + 1 sync).
NP1.0 default: 385 channels (384 neural + 1 sync).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from cerebellum_cell_classifier.io.kilosort import KilosortData

logger = logging.getLogger(__name__)

# ── Defaults (match MATLAB script) ─────────────────────────────────────────────

N_SAMPLES_DEFAULT    = 81       # window length [samples]  ~2.7 ms at 30 kHz
PEAK_SAMPLE_DEFAULT  = 40       # expected peak, 0-based  (MATLAB: peakSample=41)
N_CH_EXTRACT_DEFAULT = 8        # channels extracted per unit
MAX_SPIKES_DEFAULT   = 3_000    # cap per unit
MAX_SHIFT_DEFAULT    = 12       # realignment search half-width [samples]
N_CH_TOTAL_DEFAULT   = 385      # NP1.0 and NP2.0 both ship 385 channels


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def build_waveform_features(
    session_path: str | Path,
    ks_data: KilosortData,
    unit_ids: "np.ndarray | list[int]",
    cell_type_labels: Optional[dict] = None,
    bin_path: Optional["str | Path"] = None,
    n_channels_total: int = N_CH_TOTAL_DEFAULT,
    n_channels_extract: int = N_CH_EXTRACT_DEFAULT,
    n_samples: int = N_SAMPLES_DEFAULT,
    peak_sample: int = PEAK_SAMPLE_DEFAULT,
    max_spikes: int = MAX_SPIKES_DEFAULT,
    do_realign: bool = True,
    use_template_channel: bool = True,
    rng_seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Extract mean waveform features for a list of units.

    Parameters
    ----------
    session_path : str or Path
        Session folder containing the *.ap.bin file (auto-detected when
        bin_path is not given).
    ks_data : KilosortData
        Output of ``io.kilosort.load_kilosort()``.
    unit_ids : array-like of int
        Cluster IDs to process (any order).
    cell_type_labels : dict {unit_id → str}, optional
        Expert annotations, e.g. ``{42: 'PC', 17: 'MF'}``.
        Units absent from the dict receive the label ``'unknown'``.
        Supported strings: 'PC', 'CF', 'MLI', 'GC', 'UBC', 'MF', 'unknown'.
    bin_path : str or Path, optional
        Path to the ``*ap.bin`` raw binary.  Auto-detected when omitted.
    n_channels_total : int
        Total channels in the binary file.  NP1.0 and NP2.0 both default
        to 385 (384 neural + 1 sync).  Set to 384 if your file has no sync.
    n_channels_extract : int
        Channels extracted per unit (default: 8).
    n_samples : int
        Waveform window in samples (default: 81 ≈ 2.7 ms at 30 kHz).
    peak_sample : int
        Expected spike peak, 0-based (default: 40, centre of 81-sample window).
        MATLAB equivalent: peakSample = 41.
    max_spikes : int
        Maximum spikes used per unit (default: 3 000).
    do_realign : bool
        Realign each spike to its local peak on the main channel (default: True).
    use_template_channel : bool
        If True (default, recommended for KS4): use the 'ch' field from
        cluster_info.tsv as the main channel.
        If False: infer main channel from spike_positions.npy (requires that
        ks_data.spike_positions is not None).
    rng_seed : int
        Random seed for reproducible spike subsampling.
    verbose : bool
        Print per-unit progress.

    Returns
    -------
    dict
        ``unit_ids``         – (n_units,) int64
        ``cell_type_labels`` – (n_units,) str
        ``mean_waveforms``   – (n_units, n_ch, n_samples) float32
        ``std_waveforms``    – (n_units, n_ch, n_samples) float32
        ``main_channels``    – (n_units,) int64  hardware ch, 0-based
        ``used_channels``    – (n_units, n_ch) int64  hardware chs, main first
        ``channel_positions_used`` – (n_units, n_ch, 2) float64  µm coords of
                                     each extracted channel (for plotting)
    """
    session_path = Path(session_path)
    unit_ids     = np.asarray(unit_ids, dtype=np.int64)
    n_units      = len(unit_ids)

    # Build a fast lookup: hw_channel → row in channel_positions
    hw_to_pos_idx = {int(ch): i for i, ch in enumerate(ks_data.channel_map)}

    # ── Binary file ──────────────────────────────────────────────────────────
    if bin_path is None:
        bin_path = _find_bin_file(session_path)
    bin_path = Path(bin_path)

    n_total_samples = bin_path.stat().st_size // (2 * n_channels_total)
    raw = np.memmap(
        bin_path, dtype="int16", mode="r",
        shape=(n_total_samples, n_channels_total),
    )

    if verbose:
        dur = n_total_samples / ks_data.sample_rate
        print(f"Binary  : {bin_path.name}")
        print(f"  {n_total_samples:,} samples × {n_channels_total} ch  "
              f"({dur:.1f} s @ {ks_data.sample_rate/1e3:.0f} kHz)")
        ch_src = "template 'ch' column" if use_template_channel else "spike_positions"
        print(f"  Channel selection: {ch_src}")
        print(f"\nProcessing {n_units} units ...\n")

    # ── Output arrays ────────────────────────────────────────────────────────
    mean_wfs  = np.zeros((n_units, n_channels_extract, n_samples), dtype=np.float32)
    std_wfs   = np.zeros((n_units, n_channels_extract, n_samples), dtype=np.float32)
    main_chs  = np.full(n_units, -1, dtype=np.int64)
    used_chs  = np.full((n_units, n_channels_extract), -1, dtype=np.int64)
    ch_pos_used = np.zeros((n_units, n_channels_extract, 2), dtype=np.float64)
    n_spikes_used = np.zeros(n_units, dtype=np.int64)  # actual spike count after boundary filtering

    rng = np.random.default_rng(rng_seed)
    pad = len(str(n_units))

    for i, unit_id in enumerate(unit_ids):
        prefix = f"  [{i+1:>{pad}}/{n_units}] unit {int(unit_id)}"

        # ── Spike indices ────────────────────────────────────────────────────
        spike_idx = np.where(ks_data.spike_clusters == unit_id)[0]
        if len(spike_idx) == 0:
            logger.warning("Unit %d: no spikes — skipped.", unit_id)
            if verbose:
                print(f"{prefix}  ->  no spikes, skipped")
            continue

        if len(spike_idx) > max_spikes:
            spike_idx = rng.choice(spike_idx, size=max_spikes, replace=False)

        spike_ts = ks_data.spike_times[spike_idx]

        # ── Channel selection ────────────────────────────────────────────────
        if use_template_channel:
            if int(unit_id) not in ks_data.cluster_channels:
                logger.warning(
                    "Unit %d: no entry in cluster_channels — skipped.", unit_id
                )
                if verbose:
                    print(f"{prefix}  ->  not in cluster_info, skipped")
                continue
            main_ch_hw = ks_data.cluster_channels[int(unit_id)]
            main_pos_idx, nearest_pos_idx = _get_channels_from_main(
                main_ch_hw, ks_data.channel_map, ks_data.channel_positions,
                n_channels_extract,
            )
        else:
            if ks_data.spike_positions is None:
                raise RuntimeError(
                    "use_template_channel=False requires spike_positions.npy, "
                    "but ks_data.spike_positions is None."
                )
            spike_pos = ks_data.spike_positions[spike_idx]
            main_pos_idx, nearest_pos_idx = _get_unit_channels_from_positions(
                spike_pos, ks_data.channel_positions, n_channels_extract,
            )

        nearest_hw = ks_data.channel_map[nearest_pos_idx].astype(np.int64)
        main_ch_hw = int(ks_data.channel_map[main_pos_idx])

        # ── Extract raw waveforms ────────────────────────────────────────────
        spikes = _extract_spike_waveforms(
            raw, spike_ts, nearest_hw, n_samples, peak_sample, n_total_samples,
        )
        # spikes: (n_ch_extract, n_samples, n_valid_spikes)

        if spikes.shape[2] == 0:
            logger.warning("Unit %d: all spikes at file boundary — skipped.", unit_id)
            if verbose:
                print(f"{prefix}  ->  boundary issue, skipped")
            continue

        # ── Realign ──────────────────────────────────────────────────────────
        if do_realign:
            spikes, _ = realign_spikes(spikes, 0, peak_sample, MAX_SHIFT_DEFAULT)

        # ── Store ────────────────────────────────────────────────────────────
        n_valid = spikes.shape[2]
        mean_wfs[i]      = spikes.mean(axis=2)
        std_wfs[i]       = spikes.std(axis=2)
        main_chs[i]      = main_ch_hw
        used_chs[i]      = nearest_hw
        ch_pos_used[i]   = ks_data.channel_positions[nearest_pos_idx]
        n_spikes_used[i] = n_valid

        if verbose:
            print(f"{prefix}  ->  {spikes.shape[2]} spikes, "
                  f"main ch {main_ch_hw} "
                  f"(y={ks_data.channel_positions[main_pos_idx, 1]:.0f} µm)")

    labels = _resolve_labels(unit_ids, cell_type_labels)

    return {
        "unit_ids":               unit_ids,
        "cell_type_labels":       labels,
        "mean_waveforms":         mean_wfs,
        "std_waveforms":          std_wfs,
        "n_spikes":               n_spikes_used,   # (n_units,) actual spike count used
        "main_channels":          main_chs,
        "used_channels":          used_chs,
        "channel_positions_used": ch_pos_used,     # (n_units, n_ch, 2)  µm
    }


# ── Normalisation ────────────────────────────────────────────────────────────────

def normalize_waveforms(mean_waveforms: np.ndarray) -> np.ndarray:
    """
    Normalise each unit's multi-channel waveform:
      1. Scale so that peak-to-trough amplitude on the main channel = 1.
      2. Flip sign if the main channel's primary deflection is positive
         (paper convention: negative primary deflection).

    Parameters
    ----------
    mean_waveforms : (n_units, n_channels, n_samples) float32

    Returns
    -------
    normalised : same shape, float32
    """
    out = mean_waveforms.copy()
    for i in range(len(out)):
        primary = out[i, 0]
        amp = primary.max() - primary.min()
        if amp > 0:
            out[i] /= amp
        if abs(out[i, 0].max()) > abs(out[i, 0].min()):
            out[i] *= -1
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Core sub-routines (importable for unit tests)
# ══════════════════════════════════════════════════════════════════════════════

def realign_spikes(
    spikes: np.ndarray,
    main_channel_idx: int = 0,
    peak_sample: int = PEAK_SAMPLE_DEFAULT,
    max_shift: int = MAX_SHIFT_DEFAULT,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Realign each spike so its extremum on the main channel lands on
    ``peak_sample``.  Python equivalent of realign_spikes.m.

    Uses ``np.roll`` (wrapping), identical to MATLAB ``circshift``.

    Parameters
    ----------
    spikes          : (n_channels, n_samples, n_spikes)
    main_channel_idx: local index of the main channel (default: 0)
    peak_sample     : expected peak, 0-based (default: 40)
    max_shift       : search half-width in samples (default: 12)

    Returns
    -------
    realigned   : same shape as ``spikes``
    time_shifts : (n_spikes,) int32  — positive = shifted right
    """
    n_channels, n_samples, n_spikes = spikes.shape
    realigned   = np.empty_like(spikes)
    time_shifts = np.zeros(n_spikes, dtype=np.int32)

    s_start = peak_sample - max_shift
    s_end   = peak_sample + max_shift + 1

    mean_wf    = spikes[main_channel_idx].mean(axis=1)          # (n_samples,)
    local_peak = np.argmax(np.abs(mean_wf[s_start:s_end]))
    is_positive = mean_wf[s_start + local_peak] > 0

    for k in range(n_spikes):
        window = spikes[main_channel_idx, s_start:s_end, k]
        local_idx = int(np.argmax(window) if is_positive else np.argmin(window))
        shift = peak_sample - (s_start + local_idx)
        time_shifts[k] = shift
        realigned[:, :, k] = np.roll(spikes[:, :, k], shift, axis=1)

    return realigned, time_shifts


# ══════════════════════════════════════════════════════════════════════════════
#  Private helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_channels_from_main(
    main_ch_hw: int,
    channel_map: np.ndarray,
    channel_positions: np.ndarray,
    n_channels_extract: int,
) -> "tuple[int, np.ndarray]":
    """
    Given a known main hardware channel, return its position index and the
    ``n_channels_extract`` nearest channels (main always first).

    Parameters
    ----------
    main_ch_hw        : hardware channel number (0-based, from cluster_info 'ch')
    channel_map       : (n_active,) maps position_index → hw_channel
    channel_positions : (n_active, 2) µm
    n_channels_extract: number of channels to return

    Returns
    -------
    main_pos_idx : int  — index into channel_map / channel_positions
    nearest_idx  : (n_channels_extract,) int  — same index space, main first
    """
    hits = np.where(channel_map == main_ch_hw)[0]
    if len(hits) == 0:
        raise ValueError(
            f"Hardware channel {main_ch_hw} not found in channel_map. "
            f"channel_map range: [{channel_map.min()}, {channel_map.max()}]"
        )
    main_pos_idx = int(hits[0])

    dist = np.linalg.norm(
        channel_positions - channel_positions[main_pos_idx], axis=1
    )
    nearest_idx = np.argsort(dist)[:n_channels_extract]
    # main_pos_idx has dist=0, so it is always first after argsort

    return main_pos_idx, nearest_idx


def _get_unit_channels_from_positions(
    spike_positions: np.ndarray,    # (n_spikes, 2) µm
    channel_positions: np.ndarray,  # (n_channels, 2) µm
    n_channels_extract: int,
) -> "tuple[int, np.ndarray]":
    """
    Infer main channel from spike positions (fallback for KS2/KS3).
    See process_spike_waveforms_np.m for the original MATLAB algorithm.
    """
    diff    = spike_positions[:, np.newaxis, :] - channel_positions[np.newaxis, :, :]
    dists   = np.linalg.norm(diff, axis=2)                   # (n_spikes, n_channels)
    closest = np.argmin(dists, axis=1)                       # per-spike closest ch
    counts  = np.bincount(closest, minlength=len(channel_positions))
    main_pos_idx = int(np.argmax(counts))

    dist_to_main = np.linalg.norm(
        channel_positions - channel_positions[main_pos_idx], axis=1
    )
    nearest_idx = np.argsort(dist_to_main)[:n_channels_extract]

    return main_pos_idx, nearest_idx


def _extract_spike_waveforms(
    raw: np.memmap,
    spike_times: np.ndarray,
    channel_hw_indices: np.ndarray,
    n_samples: int,
    peak_sample: int,
    n_total_samples: int,
) -> np.ndarray:
    """
    Read a waveform window around each spike from the memory-mapped binary
    and apply the same detrending as the MATLAB ``mydetrend`` function.

    ``mydetrend`` (called by dat2spk with dflag=1) does two things per spike:
      1. DC removal  — subtract the mean of the waveform across all samples.
      2. Linear detrend — subtract the best-fit line (centered ramp projection).

    The exact Python equivalent is ``scipy.signal.detrend(type='linear')``,
    which fits and removes a least-squares line per channel, implicitly
    handling both steps in one pass.

    Boundary spikes (window goes before sample 0 or past end-of-file) are
    silently dropped.

    Returns
    -------
    spikes : (n_channels_extract, n_samples, n_valid_spikes) float32
    """
    from scipy.signal import detrend as scipy_detrend

    n_ch  = len(channel_hw_indices)
    starts = spike_times.astype(np.int64) - peak_sample
    valid  = (starts >= 0) & ((starts + n_samples) <= n_total_samples)
    valid_starts = starts[valid]
    n_valid = int(valid.sum())

    if n_valid == 0:
        return np.zeros((n_ch, n_samples, 0), dtype=np.float32)

    spikes = np.empty((n_ch, n_samples, n_valid), dtype=np.float32)
    for j, s in enumerate(valid_starts):
        # raw[s:s+n_samples] → (n_samples, n_channels_total)
        # select our channels  → (n_samples, n_ch)
        # transpose            → (n_ch, n_samples)  [matches MATLAB layout]
        wf = raw[s : s + n_samples][:, channel_hw_indices].T.astype(np.float32)

        # Equivalent of mydetrend(wf): remove DC then remove linear trend.
        # scipy detrend(type='linear') fits and subtracts a per-channel line,
        # which handles both steps in one pass (axis=1 = time axis).
        spikes[:, :, j] = scipy_detrend(wf, axis=1, type="linear")

    return spikes


def _find_bin_file(session_path: Path) -> Path:
    candidates = sorted(session_path.glob("*ap.bin"))
    if not candidates:
        raise FileNotFoundError(f"No *ap.bin file found in {session_path}")
    if len(candidates) > 1:
        logger.warning(
            "Multiple .ap.bin files in %s — using %s",
            session_path, candidates[0].name,
        )
    return candidates[0]


def _resolve_labels(
    unit_ids: np.ndarray,
    cell_type_labels: Optional[dict],
) -> np.ndarray:
    if cell_type_labels is None:
        return np.array(["unknown"] * len(unit_ids))
    return np.array([
        str(cell_type_labels.get(int(uid), "unknown"))
        for uid in unit_ids
    ])
