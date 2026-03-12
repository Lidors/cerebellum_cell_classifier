"""
ccg.py
------
Cross-correlogram (CCG) computation and CCG-based cell-type labeling.

PC / CF detection
-----------------
For each pair of nearby units, compute CCG(candidate_PC, candidate_CF).
A positive PC/CF label requires a pronounced pause in PC firing in the
2–7 ms after a CF (complex-fiber / climbing-fiber) spike.  This is visible
as suppression at *negative* lags in CCG(PC, CF), because a negative lag
of −k ms means the CF fired k ms *before* the PC reference spike, i.e.
the PC is in the post-CF silence window.

Algorithm adapted from ``PC_CS_crosscorrelogram_analysis()`` in
``CrosscorrelogramAnalysisPlugin.py`` (F. Naveros, Herzfeld lab).

MLI inhibition detection
------------------------
For each pair of nearby units, compute CCG(candidate_MLI, target).
A monosynaptic inhibition label requires suppression of the target in the
0.1–3 ms window *after* candidate spikes (positive lags).

Algorithm adapted from ``aux_crosscorrelogram_conexion_generate_inhibition()``
in ``CrosscorrelogramAnalysisPlugin.py`` (F. Naveros, Herzfeld lab).

Convention
----------
CCG(st1, st2)[center + k] = count of st2 spikes at lag k * bin_s after a
st1 reference spike.
    k > 0: st2 fires after st1.
    k < 0: st2 fires before st1.
"""

from __future__ import annotations

import numpy as np
import numba


# ══════════════════════════════════════════════════════════════════════════════
#  Numba JIT inner loop
# ══════════════════════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _ccg_matrix_engine(spike_samples: np.ndarray,
                       spike_cluster_idx: np.ndarray,
                       n_clusters: int,
                       n_half: int,
                       bin_size: int) -> np.ndarray:
    """
    All-pairs CCG/ACG counter in a single sorted-spike pass.

    Matches MATLAB CCGHeart.c exactly: integer arithmetic, same bin formula.
    For every ordered pair of spikes (j before i), increments BOTH directions
    of the CCG matrix simultaneously — one pass handles all pairs.

    Parameters
    ----------
    spike_samples     : (n,) int64, **sorted** merged spike times in samples
    spike_cluster_idx : (n,) int64, 0-based cluster index per spike
    n_clusters        : int — number of distinct clusters
    n_half            : int — bins on each side of zero lag
    bin_size          : int — bin width in samples

    Returns
    -------
    counts : (2*n_half+1, n_clusters, n_clusters) int64
        counts[b, i, j] = times cluster j fired at lag b relative to cluster i.
        b = n_half  → lag 0 (center)
        b > n_half  → positive lag (cluster j fires after cluster i)
        b < n_half  → negative lag (cluster j fires before cluster i)
        Diagonal counts[:, i, i] = ACG (0-lag center NOT zeroed here).
    """
    n_bins    = 2 * n_half + 1
    counts    = np.zeros((n_bins, n_clusters, n_clusters), dtype=np.int64)
    max_lag   = n_half * bin_size + bin_size // 2   # matches MATLAB furthest
    bin_size2 = 2 * bin_size
    n         = len(spike_samples)

    j_lo = 0
    for i in range(n):
        # Advance j_lo: earliest j where s[i] - s[j] <= max_lag
        while j_lo < i and spike_samples[i] - spike_samples[j_lo] > max_lag:
            j_lo += 1
        ci = spike_cluster_idx[i]
        for j in range(j_lo, i):
            diff = spike_samples[i] - spike_samples[j]  # in [0, max_lag] (sorted)
            cj = spike_cluster_idx[j]

            # Forward: ref=cj, target=ci at positive lag.
            # Matches MATLAB forward pass: excluded when diff == max_lag (abs >= furthest).
            b_fwd = (2 * diff + bin_size) // bin_size2 + n_half
            if b_fwd < n_bins:
                counts[b_fwd, cj, ci] += 1

            # Backward: ref=ci, target=cj at negative lag.
            # Matches MATLAB: b_bwd = n_half + floor(0.5 - diff/bin_size)
            #                        = n_half + (bin_size - 2*diff) // (2*bin_size)
            # MUST be outside the b_fwd check — MATLAB backward uses > (includes diff==max_lag).
            b_bwd = n_half + (bin_size - 2 * diff) // bin_size2
            counts[b_bwd, ci, cj] += 1

    return counts


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _smooth(arr: np.ndarray, k: int) -> np.ndarray:
    """
    Boxcar smooth with kernel of size k (nearest odd integer).
    Border correction matches the Herzfeld-lab plugin's smooth() function.
    """
    N = int(k / 2) * 2 + 1
    if N < 3:
        return arr.astype(np.float64)
    box = np.ones(N) / N
    y = np.convolve(arr.astype(np.float64), box, mode="same")
    half = N // 2
    for i in range(half):                          # fix left border
        y[i] = np.mean(arr[: 2 * i + 1])
    for i in range(half):                          # fix right border
        y[len(y) - 1 - i] = np.mean(arr[len(arr) - 2 * i - 1 :])
    return y


def _baseline_stats(ccg: np.ndarray, n_half: int,
                    n_outer: int = 20,
                    side: str = "both") -> tuple[float, float]:
    """
    Estimate baseline mean and std from the ``n_outer`` outermost bins on
    each tail (or one tail) of the CCG.

    With n_half = 100 (±10 ms at 0.1 ms bins) and n_outer = 20, the
    baseline covers lags ±8–10 ms, which does not overlap any analysis
    window (PC/CF pause: −7 to −2 ms; MLI: ±0.1 to ±3 ms).

    Parameters
    ----------
    side : 'both', 'left', or 'right'
        'both'  — use outer bins from both tails (default, good for MLI)
        'right' — use only the positive-lag tail (good for PC/CF, where
                   the pause at negative lags contaminates the left tail)
        'left'  — use only the negative-lag tail
    """
    n_total = 2 * n_half + 1
    if side == "right":
        outer = ccg[n_total - n_outer:]
    elif side == "left":
        outer = ccg[:n_outer]
    else:
        outer = np.concatenate([ccg[:n_outer], ccg[n_total - n_outer:]])
    if len(outer) == 0:
        return 0.0, 0.0
    return float(np.mean(outer)), float(np.std(outer))


# ══════════════════════════════════════════════════════════════════════════════
#  Public CCG computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_ccg_matrix(
    spike_times_s:  np.ndarray,
    spike_clusters: np.ndarray,
    cluster_ids,
    lag_ms:         float = 20.0,
    bin_ms:         float = 0.2,
    sample_rate:    float = 30_000.0,
    normalization:  str   = "rate",
) -> tuple[np.ndarray, np.ndarray]:
    """
    CCG / ACG matrix for a selected set of clusters.

    The core function of this module — mirrors MATLAB's CCG() interface.
    All other CCG/ACG functions are thin wrappers around this one.

    Parameters
    ----------
    spike_times_s  : (n_total,) float64 — all spike times in **seconds**
    spike_clusters : (n_total,) int64   — cluster ID per spike
    cluster_ids    : array-like of int  — which clusters to compute
                     Pass one cluster  → returns [n_bins × 1 × 1] (ACG).
                     Pass two clusters → returns [n_bins × 2 × 2] (ACG + CCG).
                     Pass N clusters   → returns [n_bins × N × N] (full matrix).
    lag_ms         : half-window in ms (default 20)
    bin_ms         : bin size in ms (default 0.2)
    sample_rate    : Hz (default 30 000)
    normalization  : 'rate'  → conditional firing rate [Hz]
                               = counts / (n_ref_spikes × bin_s)  (float64)
                     'count' → raw spike counts (int64)

    Returns
    -------
    cc   : (n_bins, n_clusters, n_clusters)
               cc[:, i, i] = ACG of cluster_ids[i]  (0-lag center zeroed)
               cc[:, i, j] = CCG(cluster_ids[i] → cluster_ids[j])
    t_ms : (n_bins,) float64 — lag axis in ms
    """
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    n_clusters  = len(cluster_ids)
    n_half      = int(lag_ms / bin_ms)
    n_bins      = 2 * n_half + 1
    t_ms        = np.arange(-n_half, n_half + 1, dtype=np.float64) * bin_ms
    bin_size    = int(round(bin_ms / 1000.0 * sample_rate))
    bin_s       = bin_ms / 1000.0

    spike_clusters = np.asarray(spike_clusters, dtype=np.int64)
    spike_times_s  = np.asarray(spike_times_s,  dtype=np.float64)

    # Filter to selected clusters
    mask     = np.isin(spike_clusters, cluster_ids)
    clu_filt = spike_clusters[mask]
    st_filt  = spike_times_s [mask]

    # Sort by spike time
    order    = np.argsort(st_filt, kind="stable")
    st_filt  = st_filt [order]
    clu_filt = clu_filt[order]

    # Convert to integer samples
    st_int = np.round(st_filt * sample_rate).astype(np.int64)

    # Map cluster IDs → 0-based indices for the engine
    clu_idx = np.empty(len(clu_filt), dtype=np.int64)
    for k in range(n_clusters):
        clu_idx[clu_filt == cluster_ids[k]] = k

    # Spike count per cluster (needed for rate normalisation)
    n_spikes = np.zeros(n_clusters, dtype=np.int64)
    for k in range(n_clusters):
        n_spikes[k] = int(np.sum(clu_idx == k))

    if len(st_int) == 0:
        empty = np.zeros((n_bins, n_clusters, n_clusters), dtype=np.float64)
        return empty, t_ms

    counts = _ccg_matrix_engine(st_int, clu_idx, n_clusters, n_half, bin_size)

    # Zero ACG center bin (0-lag self-coincidence artefact)
    for k in range(n_clusters):
        counts[n_half, k, k] = 0

    if normalization == "count":
        return counts, t_ms

    # 'rate': divide each row i by n_spikes[i] * bin_s → Hz
    cc = np.zeros((n_bins, n_clusters, n_clusters), dtype=np.float64)
    for i in range(n_clusters):
        if n_spikes[i] > 0:
            cc[:, i, :] = counts[:, i, :].astype(np.float64) / (n_spikes[i] * bin_s)
    return cc, t_ms


def compute_ccg(
    st1_s: np.ndarray,
    st2_s: np.ndarray,
    lag_ms: float = 20.0,
    bin_ms: float = 0.2,
    sample_rate: "float | None" = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Cross-correlogram between two spike trains.

    Thin wrapper around ``compute_ccg_matrix`` for the common two-train case.

    Parameters
    ----------
    st1_s       : spike times in **seconds** for the reference (trigger) train
    st2_s       : spike times in **seconds** for the target train
    lag_ms      : half-window in ms (default 20)
    bin_ms      : bin size in ms (default 0.2)
    sample_rate : recording sample rate in Hz (required)

    Returns
    -------
    counts : (2*n_half+1,) int64 — raw spike counts per bin
    t_ms   : (2*n_half+1,) float64 — lag axis in ms
    n_half : int — number of bins per side (center index)
    """
    if sample_rate is None:
        raise ValueError(
            "sample_rate is required. Pass the recording sample rate (e.g. 30_000.0) "
            "so that spike times are binned with exact integer arithmetic."
        )

    n_half = int(lag_ms / bin_ms)
    t_ms   = np.arange(-n_half, n_half + 1, dtype=np.float64) * bin_ms

    st1 = np.sort(np.asarray(st1_s, dtype=np.float64))
    st2 = np.sort(np.asarray(st2_s, dtype=np.float64))

    if len(st1) == 0 or len(st2) == 0:
        return np.zeros(2 * n_half + 1, dtype=np.int64), t_ms, n_half

    # Merge with temporary cluster IDs: 0 = st1, 1 = st2
    st_all  = np.concatenate([st1, st2])
    clu_all = np.concatenate([
        np.zeros(len(st1), dtype=np.int64),
        np.ones( len(st2), dtype=np.int64),
    ])

    cc, _ = compute_ccg_matrix(
        st_all, clu_all, np.array([0, 1], dtype=np.int64),
        lag_ms=lag_ms, bin_ms=bin_ms, sample_rate=sample_rate,
        normalization="count",
    )
    return cc[:, 0, 1], t_ms, n_half


# ══════════════════════════════════════════════════════════════════════════════
#  PC / CF pause analysis
# ══════════════════════════════════════════════════════════════════════════════

def pc_cf_pause_metrics(
    ccg_smooth: np.ndarray,
    n_half:     int,
    bin_ms:     float = 0.1,
) -> dict:
    """
    Compute PC/CF (complex-fiber) pause metrics from a smoothed CCG(PC, CF).

    The pause window (−7 to −2 ms) captures PC simple-spike suppression
    in the post-CF period.  In CCG(PC, CF) a negative lag −k means the CF
    fired k ms *before* the PC reference spike, i.e. the PC is k ms into
    the post-CF silence.

    Adapted from ``PC_CS_crosscorrelogram_analysis()`` in the Herzfeld-lab
    ``CrosscorrelogramAnalysisPlugin.py``.

    Parameters
    ----------
    ccg_smooth : (2*n_half+1,) float — smoothed CCG counts
    n_half     : center bin index (= int(lag_ms / bin_ms))
    bin_ms     : bin size in ms (default 0.1)

    Returns
    -------
    dict
        variability     : baseline CV (std / mean)
        pause_ratio     : lower-30th-pct of pause window  / baseline mean
        no_pause_ratio  : lower-30th-pct of rebound window / baseline mean
        min_pause_ratio : absolute minimum of pause window / baseline mean
        asym            : no_pause_value / pause_value  (asymmetry ratio)
        combined        : variability + pause_ratio
        mean_baseline   : estimated baseline mean
        is_pc_cf        : bool — True if all criteria are satisfied
    """
    tbin_s = bin_ms / 1000.0
    # Use only the POSITIVE (right) tail for baseline, because the pause
    # at negative lags makes the left tail unreliable for a real PC/CF pair.
    mean_val, std_val = _baseline_stats(ccg_smooth, n_half, side="right")

    result: dict = dict(
        variability=np.nan, pause_ratio=np.nan, no_pause_ratio=np.nan,
        min_pause_ratio=np.nan, asym=np.nan, combined=np.nan,
        mean_baseline=mean_val, is_pc_cf=False,
    )

    if mean_val <= 0.0:
        return result

    variability = std_val / mean_val
    result["variability"] = variability

    if variability >= 0.5:
        return result

    # Pause window  : −7 ms to −2 ms  (70 bins and 20 bins at 0.1 ms)
    # Rebound window: +2 ms to +7 ms
    n_center = int(round(0.007 / tbin_s))   # 70
    n_offset = int(round(0.002 / tbin_s))   # 20

    i_pause_lo = n_half - n_center           # 30
    i_pause_hi = n_half - n_offset           # 80
    i_no_lo    = n_half + n_offset           # 120
    i_no_hi    = n_half + n_center           # 170

    if i_pause_lo < 0 or i_no_hi > len(ccg_smooth):
        return result

    pause_sorted    = np.sort(ccg_smooth[i_pause_lo : i_pause_hi])
    no_pause_sorted = np.sort(ccg_smooth[i_no_lo    : i_no_hi   ])

    if len(pause_sorted) == 0 or len(no_pause_sorted) == 0:
        return result

    n_take       = max(1, int(round(len(pause_sorted) * 0.3)))
    pause_val    = float(np.mean(pause_sorted[:n_take]))
    no_pause_val = float(np.mean(no_pause_sorted[:n_take]))
    min_pause    = float(pause_sorted[0])

    result["min_pause_ratio"] = min_pause    / mean_val
    result["pause_ratio"]     = pause_val    / mean_val
    result["no_pause_ratio"]  = no_pause_val / mean_val
    result["asym"]            = no_pause_val / pause_val if pause_val > 0 else np.inf
    result["combined"]        = variability  + pause_val / mean_val

    result["is_pc_cf"] = (
        min_pause    / mean_val < 0.15
        and pause_val    / mean_val < 0.35
        and (no_pause_val / pause_val > 5.0 if pause_val > 0 else no_pause_val > 0)
        and no_pause_val / mean_val > 0.35
        and variability  + pause_val / mean_val < 0.70
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MLI inhibition analysis
# ══════════════════════════════════════════════════════════════════════════════

def mli_inhibition_metrics(
    ccg_smooth:           np.ndarray,
    n_half:               int,
    bin_ms:               float = 0.1,
    variance_ratio:       float = 1.05,
    max_queue_variability: float = 0.3,
) -> dict:
    """
    Detect monosynaptic inhibition in CCG(MLI, target).

    Criteria (default thresholds):
    - Baseline variability (CV) < ``max_queue_variability`` (0.3)
    - Mean of right window (+0.1 to +3 ms) < 1/``variance_ratio`` × baseline
      (target is suppressed after MLI spike)
    - Mean of left window  (−3 to −0.1 ms) > 1/``variance_ratio`` × baseline
      (pre-spike period is at baseline, confirming causal direction)
    - Lower 10 bins of right window: 0.2–0.85 × baseline
      (partial, not silent, suppression)
    - Lower 10 bins of left  window: > 0.85 × baseline

    Adapted from ``aux_crosscorrelogram_conexion_generate_inhibition()``
    in the Herzfeld-lab ``CrosscorrelogramAnalysisPlugin.py``.

    Parameters
    ----------
    ccg_smooth            : (2*n_half+1,) float — smoothed CCG(MLI, target)
    n_half                : center bin index
    bin_ms                : bin size in ms
    variance_ratio        : asymmetry threshold (default 1.05)
    max_queue_variability : baseline CV ceiling (default 0.3)

    Returns
    -------
    dict
        queue_variability : baseline CV
        left_dev          : mean(left window) / baseline
        right_dev         : mean(right window) / baseline
        min_left_norm     : lower-10-bin mean of left  / baseline
        min_right_norm    : lower-10-bin mean of right / baseline
        mean_baseline     : estimated baseline mean
        is_inhibitory     : bool
    """
    mean_val, std_val = _baseline_stats(ccg_smooth, n_half)

    result: dict = dict(
        queue_variability=np.nan, left_dev=np.nan, right_dev=np.nan,
        min_left_norm=np.nan, min_right_norm=np.nan,
        mean_baseline=mean_val, is_inhibitory=False,
    )

    if mean_val <= 0.0:
        return result

    q_var = std_val / mean_val
    result["queue_variability"] = q_var

    if q_var >= max_queue_variability:
        return result

    # ±3 ms analysis windows  (30 bins at 0.1 ms, matches plugin hardcode)
    n_win = 30
    left_sl  = ccg_smooth[n_half - n_win : n_half    ]  # −3 ms to −0.1 ms
    right_sl = ccg_smooth[n_half + 1     : n_half + n_win + 1]  # +0.1 ms to +3 ms

    if len(left_sl) < 10 or len(right_sl) < 10:
        return result

    left_dev  = float(np.mean(left_sl))  / mean_val
    right_dev = float(np.mean(right_sl)) / mean_val

    n_min = min(10, len(left_sl), len(right_sl))
    min_left_norm  = float(np.mean(np.sort(left_sl )[:n_min])) / mean_val
    min_right_norm = float(np.mean(np.sort(right_sl)[:n_min])) / mean_val

    result["left_dev"]       = left_dev
    result["right_dev"]      = right_dev
    result["min_left_norm"]  = min_left_norm
    result["min_right_norm"] = min_right_norm

    thresh = 1.0 / variance_ratio
    result["is_inhibitory"] = (
        right_dev      < thresh
        and left_dev       > thresh
        and 0.2 < min_right_norm < 0.85
        and min_left_norm > 0.85
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Main labeling pipeline
# ══════════════════════════════════════════════════════════════════════════════

def build_ccg_labels(
    unit_ids:           np.ndarray,
    spike_times:        np.ndarray,      # samples (int64)
    spike_clusters:     np.ndarray,      # int64
    unit_positions_um:  np.ndarray,      # (N, 2) um, main-channel [x, y] per unit
    initial_labels:     "np.ndarray | None" = None,   # (N,) str
    sample_rate:        float = 30_000.0,
    max_distance_um:    float = 200.0,
    min_pc_fr_hz:       float = 30.0,
    lag_ms:             float = 20.0,
    bin_ms:             float = 0.2,
    smooth_k_pc_cf:     int   = 11,
    smooth_k_mli:       int   = 3,
    variance_ratio:     float = 1.05,
    max_queue_variability: float = 0.3,
    min_spikes:         int   = 100,
    verbose:            bool  = True,
) -> dict:
    """
    Auto-label units as PC, CF, or MLI using pairwise CCG analysis.

    All CCG arrays for tested pairs are returned so they can be saved
    for later inspection in the pair viewer.

    Parameters
    ----------
    unit_ids          : (N,) int — cluster IDs to label
    spike_times       : all spike times in **samples** (from spike_times.npy)
    spike_clusters    : cluster ID per spike (from spike_clusters.npy)
    unit_positions_um : (N, 2) float — [x, y] um of main channel per unit
    initial_labels    : (N,) str — starting labels; None -> all 'unknown'
    sample_rate       : Hz (default 30 000)
    max_distance_um   : only test pairs closer than this (default 200 um)
    min_pc_fr_hz      : minimum mean FR for a PC candidate (default 30 Hz)
    lag_ms            : CCG half-window in ms (default 10)
    bin_ms            : CCG bin size in ms (default 0.1)
    smooth_k_pc_cf    : smoothing kernel for PC/CF CCG (default 11)
    smooth_k_mli      : smoothing kernel for MLI CCG (default 3)
    variance_ratio     : MLI asymmetry threshold (default 1.1)
    max_queue_variability : MLI baseline CV ceiling (default 0.3)
    min_spikes        : skip units with fewer spikes than this (default 100)
    verbose           : print progress

    Returns
    -------
    dict
        labels          : (N,) str — updated labels (expert preserved)
        ccg_auto_labels : (N,) str — pure CCG detection labels (independent
                          of expert labels; always starts from 'unknown')
        pc_cf_pairs     : list of (pc_uid, cf_uid) tuples detected
        mli_units       : list of uid detected as MLI
        pair_unit_ids   : (P, 2) int64 — unit IDs for each unordered pair
        pair_ccgs       : (P, n_bins) float32 — CCG(uid_A -> uid_B) counts
        pair_dists      : (P,) float32 — distance in um for each pair
        pair_types      : (P,) str — 'pc_cf', 'mli', 'none'
        pair_scores     : (P,) float32 — score for sorting (higher = stronger)
        ccg_t_ms        : (n_bins,) float64 — lag axis in ms
    """
    unit_ids  = np.asarray(unit_ids, dtype=np.int64)
    n_units   = len(unit_ids)
    positions = np.asarray(unit_positions_um, dtype=np.float64)
    n_half    = int(lag_ms / bin_ms)
    n_bins    = 2 * n_half + 1

    # Two label arrays:
    #   labels          — starts from initial_labels, only non-expert updated
    #   ccg_auto_labels — pure CCG result, starts from all 'unknown'
    if initial_labels is None:
        labels = np.array(["unknown"] * n_units, dtype=object)
    else:
        labels = np.array(initial_labels, dtype=object)

    ccg_auto_labels = np.array(["unknown"] * n_units, dtype=object)
    expert     = (labels != "unknown")
    uid_to_idx = {int(uid): i for i, uid in enumerate(unit_ids)}

    # ── Pre-fetch spike trains ────────────────────────────────────────────────
    if verbose:
        print("  Fetching spike trains ...")

    sts:      dict[int, np.ndarray] = {}
    mean_frs: dict[int, float]      = {}

    for uid in unit_ids:
        uid_i = int(uid)
        st = spike_times[spike_clusters == uid_i].astype(np.float64) / sample_rate
        st = np.sort(st)
        sts[uid_i]      = st
        dur             = (st[-1] - st[0]) if len(st) > 1 else 1.0
        mean_frs[uid_i] = len(st) / dur if dur > 0 else 0.0

    # ── Build candidate pair list (unordered, i < j) ─────────────────────────
    pairs: list[tuple[int, int, float]] = []
    for i in range(n_units):
        for j in range(i + 1, n_units):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            if dist <= max_distance_um:
                pairs.append((i, j, dist))

    n_pairs = len(pairs)
    if verbose:
        print(f"  {n_pairs} unit pairs within {max_distance_um:.0f} um "
              f"(note: first call compiles Numba JIT ~1-2 s) ...")

    # ── Allocate pair-level storage ──────────────────────────────────────────
    all_pair_ids   = np.zeros((n_pairs, 2), dtype=np.int64)
    all_pair_ccgs  = np.zeros((n_pairs, n_bins), dtype=np.float32)
    all_pair_dists = np.zeros(n_pairs, dtype=np.float32)
    all_pair_types = np.array(["none"] * n_pairs, dtype=object)
    all_pair_scores = np.zeros(n_pairs, dtype=np.float32)

    pc_cf_pairs:  list[tuple[int, int]] = []
    mli_cands:    set[int]              = set()
    mli_units:    list[int]             = []

    # ── Compute CCGs and analyse all pairs ───────────────────────────────────
    if verbose:
        print("  Computing CCGs and analysing pairs ...")

    for p_idx, (i, j, dist) in enumerate(pairs):
        uid_i = int(unit_ids[i])
        uid_j = int(unit_ids[j])
        all_pair_ids[p_idx]   = [uid_i, uid_j]
        all_pair_dists[p_idx] = dist

        st_i = sts[uid_i]
        st_j = sts[uid_j]

        # Skip pairs with too few spikes
        if len(st_i) < min_spikes or len(st_j) < min_spikes:
            continue

        # Merge the two spike trains and call the unified engine.
        # cc[:, 0, 1] = CCG(uid_i → uid_j)
        # cc[:, 1, 0] = CCG(uid_j → uid_i)
        st_pair  = np.concatenate([st_i, st_j])
        clu_pair = np.concatenate([
            np.full(len(st_i), uid_i, dtype=np.int64),
            np.full(len(st_j), uid_j, dtype=np.int64),
        ])
        cc, _ = compute_ccg_matrix(
            st_pair, clu_pair,
            np.array([uid_i, uid_j], dtype=np.int64),
            lag_ms=lag_ms, bin_ms=bin_ms,
            sample_rate=sample_rate,
            normalization="count",
        )

        all_pair_ccgs[p_idx] = cc[:, 0, 1].astype(np.float32)

        counts_f = cc[:, 0, 1].astype(np.float64)  # CCG(uid_i → uid_j)
        counts_r = cc[:, 1, 0].astype(np.float64)  # CCG(uid_j → uid_i)

        # ── PC / CF check (both directions) ──────────────────────────────
        sm_fwd_pc = _smooth(counts_f, smooth_k_pc_cf)
        sm_rev_pc = _smooth(counts_r, smooth_k_pc_cf)

        m_fwd = pc_cf_pause_metrics(sm_fwd_pc, n_half, bin_ms)
        m_rev = pc_cf_pause_metrics(sm_rev_pc, n_half, bin_ms)

        fwd_pc = m_fwd["is_pc_cf"] and mean_frs[uid_i] >= min_pc_fr_hz
        rev_pc = m_rev["is_pc_cf"] and mean_frs[uid_j] >= min_pc_fr_hz

        if fwd_pc:
            ccg_auto_labels[i] = "PC"
            ccg_auto_labels[j] = "CF"
            if not expert[i]: labels[i] = "PC"
            if not expert[j]: labels[j] = "CF"
            pc_cf_pairs.append((uid_i, uid_j))
            score = m_fwd["asym"] if np.isfinite(m_fwd["asym"]) else 999.0
            if score > all_pair_scores[p_idx]:
                all_pair_types[p_idx]  = "pc_cf"
                all_pair_scores[p_idx] = score
            if verbose:
                print(f"    PC/CF: unit {uid_i} (FR={mean_frs[uid_i]:.1f} Hz)"
                      f" -> unit {uid_j}  dist={dist:.0f} um  "
                      f"pause={m_fwd['pause_ratio']:.2f}  "
                      f"asym={m_fwd['asym']:.1f}")

        if rev_pc:
            ccg_auto_labels[j] = "PC"
            ccg_auto_labels[i] = "CF"
            if not expert[j]: labels[j] = "PC"
            if not expert[i]: labels[i] = "CF"
            pc_cf_pairs.append((uid_j, uid_i))
            score = m_rev["asym"] if np.isfinite(m_rev["asym"]) else 999.0
            if score > all_pair_scores[p_idx]:
                all_pair_types[p_idx]  = "pc_cf"
                all_pair_scores[p_idx] = score
            if verbose:
                print(f"    PC/CF: unit {uid_j} (FR={mean_frs[uid_j]:.1f} Hz)"
                      f" -> unit {uid_i}  dist={dist:.0f} um  "
                      f"pause={m_rev['pause_ratio']:.2f}  "
                      f"asym={m_rev['asym']:.1f}")

        # ── MLI inhibition check (both directions) ───────────────────────
        sm_fwd_ml = _smooth(counts_f, smooth_k_mli)
        sm_rev_ml = _smooth(counts_r, smooth_k_mli)

        ml_fwd = mli_inhibition_metrics(sm_fwd_ml, n_half, bin_ms,
                                        variance_ratio, max_queue_variability)
        ml_rev = mli_inhibition_metrics(sm_rev_ml, n_half, bin_ms,
                                        variance_ratio, max_queue_variability)

        # fwd: uid_i = MLI (source), uid_j = target (inhibited)
        if ml_fwd["is_inhibitory"] and mean_frs[uid_j] >= min_pc_fr_hz:
            mli_cands.add(uid_i)
            if all_pair_types[p_idx] == "none":
                score = abs(1.0 - ml_fwd["right_dev"]) if np.isfinite(ml_fwd["right_dev"]) else 0.0
                all_pair_types[p_idx]  = "mli"
                all_pair_scores[p_idx] = score
            if verbose:
                print(f"    MLI: unit {uid_i} -> unit {uid_j}  "
                      f"dist={dist:.0f} um  "
                      f"right={ml_fwd['right_dev']:.2f}  "
                      f"left={ml_fwd['left_dev']:.2f}  "
                      f"tgt_FR={mean_frs[uid_j]:.1f}")

        # rev: uid_j = MLI (source), uid_i = target (inhibited)
        if ml_rev["is_inhibitory"] and mean_frs[uid_i] >= min_pc_fr_hz:
            mli_cands.add(uid_j)
            if all_pair_types[p_idx] == "none":
                score = abs(1.0 - ml_rev["right_dev"]) if np.isfinite(ml_rev["right_dev"]) else 0.0
                all_pair_types[p_idx]  = "mli"
                all_pair_scores[p_idx] = score
            if verbose:
                print(f"    MLI: unit {uid_j} -> unit {uid_i}  "
                      f"dist={dist:.0f} um  "
                      f"right={ml_rev['right_dev']:.2f}  "
                      f"left={ml_rev['left_dev']:.2f}  "
                      f"tgt_FR={mean_frs[uid_i]:.1f}")

    # Apply MLI auto-labels — never relabel PC or CF
    for uid in mli_cands:
        idx = uid_to_idx[uid]
        if ccg_auto_labels[idx] not in ("PC", "CF"):
            ccg_auto_labels[idx] = "MLI"
        if not expert[idx] and labels[idx] not in ("PC", "CF"):
            labels[idx] = "MLI"
            mli_units.append(uid)

    # Build t_ms axis (same for all pairs)
    ccg_t_ms = np.arange(-n_half, n_half + 1, dtype=np.float64) * bin_ms

    if verbose:
        n_pc  = int((labels == "PC").sum())
        n_cf  = int((labels == "CF").sum())
        n_mli = int((labels == "MLI").sum())
        n_det = int((all_pair_types != "none").sum())
        print(f"  CCG labeling done:  {n_det} detected pair(s) of {n_pairs}  "
              f"-> {n_pc} PC, {n_cf} CF, {n_mli} MLI")

    return {
        "labels":          labels,
        "ccg_auto_labels": ccg_auto_labels,
        "pc_cf_pairs":     pc_cf_pairs,
        "mli_units":       mli_units,
        "pair_unit_ids":   all_pair_ids,        # (P, 2) int64
        "pair_ccgs":       all_pair_ccgs,       # (P, n_bins) float32
        "pair_dists":      all_pair_dists,      # (P,) float32
        "pair_types":      all_pair_types,      # (P,) str
        "pair_scores":     all_pair_scores,     # (P,) float32
        "ccg_t_ms":        ccg_t_ms,            # (n_bins,) float64
    }
