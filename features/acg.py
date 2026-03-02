"""
acg.py
------
Compute 1-D and 3-D autocorrelograms (ACGs) from spike trains.

1D ACG  : classic lag histogram, normalised to conditional firing rate [Hz].
           Direct Python equivalent of CCG.m / CCGHeart.c (Ken Harris / Nate lab)
           with 'hz' normalisation.

3D ACG  : firing-rate-conditioned ACG with log-scaled lag axis.
           Matches the approach used in the C4 / NeuroPyxels cerebellar classifier
           (Beau et al., Nat Neurosci 2023 / npyx corr.crosscorr_vs_firing_rate +
           corr.convert_acg_log):

           Step 1 — compute a 2D matrix (linear lag × FR-quantile bin) using the
                    same Numba sliding-window engine as the 1D ACG.  Each column
                    contains the ACG of reference spikes whose instantaneous firing
                    rate fell in a given quantile bin.

           Step 2 — re-sample the lag axis from linear to log-spaced bins.
                    This compresses the slow timescale and zooms into the fast
                    structure near lag = 0, producing the characteristic
                    "3D ACG" heatmap shape used for classification.

The inner counting loops are JIT-compiled with Numba (C-level speed, no manual
compilation required — Numba compiles on first call and caches to disk).

Note: the very first call to each engine function takes ~1-2 s for JIT
compilation.  All subsequent calls are fast (cached to disk).
"""

from __future__ import annotations

import numpy as np
import numba


# ══════════════════════════════════════════════════════════════════════════════
#  Numba JIT inner loops
# ══════════════════════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _acg_1d_engine(spike_times_s, half_bins, bin_size_s):
    """
    Bidirectional sliding-window ACG counter.
    Equivalent to CCGHeart.c for a single spike train.

    Parameters
    ----------
    spike_times_s : (n,) float64, **sorted**, spike times in seconds
    half_bins     : int   — bins on each side of zero lag
    bin_size_s    : float — bin width in seconds

    Returns
    -------
    counts : (2*half_bins + 1,) int64
    """
    n_bins  = 2 * half_bins + 1
    counts  = np.zeros(n_bins, dtype=np.int64)
    n       = len(spike_times_s)
    max_lag = half_bins * bin_size_s

    j_start = 0
    for i in range(n):
        while j_start < n and spike_times_s[i] - spike_times_s[j_start] > max_lag:
            j_start += 1
        for j in range(j_start, n):
            if j == i:
                continue
            diff = spike_times_s[j] - spike_times_s[i]
            if diff > max_lag:
                break
            b = int(diff / bin_size_s + 0.5) + half_bins
            if 0 <= b < n_bins:
                counts[b] += 1
    return counts


@numba.njit(cache=True)
def _acg_3d_engine(spike_times_s, fr_per_spike, fr_bin_edges,
                   half_bins, bin_size_s):
    """
    Firing-rate-conditioned ACG counter (linear lag axis).

    For every reference spike that has a valid instantaneous firing rate
    (i.e. not the first spike), accumulate co-spikes into a 2-D array
    indexed by (lag_bin, fr_bin).

    Parameters
    ----------
    spike_times_s : (n,) float64, sorted, seconds
    fr_per_spike  : (n,) float64 — instantaneous FR [Hz] = 1/ISI_preceding;
                    NaN for the first spike (no preceding ISI)
    fr_bin_edges  : (n_fr_bins+1,) float64 — FR quantile bin edges [Hz]
    half_bins     : int
    bin_size_s    : float

    Returns
    -------
    counts    : (2*half_bins+1, n_fr_bins) int64
    ref_count : (n_fr_bins,) int64 — reference spikes per FR bin
    """
    n_lag_bins = 2 * half_bins + 1
    n_fr_bins  = len(fr_bin_edges) - 1
    counts     = np.zeros((n_lag_bins, n_fr_bins), dtype=np.int64)
    ref_count  = np.zeros(n_fr_bins,               dtype=np.int64)
    n          = len(spike_times_s)
    max_lag    = half_bins * bin_size_s

    j_start = 0
    for i in range(n):
        fr_i = fr_per_spike[i]
        if np.isnan(fr_i):
            continue

        # Binary-search for FR bin (edges are sorted)
        fr_bin = n_fr_bins - 1          # default: last bin
        for b in range(n_fr_bins):
            if fr_i < fr_bin_edges[b + 1]:
                fr_bin = b
                break
        ref_count[fr_bin] += 1

        while j_start < n and spike_times_s[i] - spike_times_s[j_start] > max_lag:
            j_start += 1

        for j in range(j_start, n):
            if j == i:
                continue
            diff = spike_times_s[j] - spike_times_s[i]
            if diff > max_lag:
                break
            lag_bin = int(diff / bin_size_s + 0.5) + half_bins
            if 0 <= lag_bin < n_lag_bins:
                counts[lag_bin, fr_bin] += 1

    return counts, ref_count


# ══════════════════════════════════════════════════════════════════════════════
#  Log-lag conversion  (equivalent to npyx corr.convert_acg_log)
# ══════════════════════════════════════════════════════════════════════════════

def _convert_lag_to_log(
    acg_linear: np.ndarray,
    bin_ms:     float,
    n_log_bins: int   = 100,
    log_start_ms: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Re-sample the lag axis of a 2-D ACG from linear to log-spaced bins.

    Parameters
    ----------
    acg_linear   : (n_lag_linear, n_fr_bins) — linear-lag ACG [Hz]
    bin_ms       : linear bin size in ms
    n_log_bins   : number of log-spaced bins for the positive half (default 100)
    log_start_ms : first log bin in ms (default 1.0)

    Returns
    -------
    acg_log : (2*n_log_bins + 1, n_fr_bins) — log-lag ACG [Hz]
    t_log   : (2*n_log_bins + 1,) — lag axis in ms (symmetric around 0)
    """
    half       = (acg_linear.shape[0] - 1) // 2
    n_fr_bins  = acg_linear.shape[1]
    lag_max_ms = half * bin_ms

    # Log-spaced positive lag axis
    t_log_pos  = np.logspace(
        np.log10(log_start_ms), np.log10(lag_max_ms), n_log_bins
    )
    # Linear positive lag axis (skip lag-0 bin)
    t_lin_pos  = np.arange(1, half + 1, dtype=np.float64) * bin_ms
    pos_acg    = acg_linear[half + 1:]   # (half, n_fr_bins) — positive lags only

    acg_log = np.zeros((2 * n_log_bins + 1, n_fr_bins), dtype=np.float64)
    for b in range(n_fr_bins):
        interp = np.interp(t_log_pos, t_lin_pos, pos_acg[:, b])
        acg_log[:n_log_bins, b] = interp[::-1]   # negative lags (mirrored)
        # centre bin stays 0
        acg_log[n_log_bins + 1:, b] = interp     # positive lags

    t_log = np.concatenate([-t_log_pos[::-1], [0.0], t_log_pos])
    return acg_log, t_log


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def compute_acg(
    spike_times_s: np.ndarray,
    lag_ms:  float = 500.0,
    bin_ms:  float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1-D autocorrelogram for a single unit.

    Parameters
    ----------
    spike_times_s : spike times in **seconds** (any order)
    lag_ms        : half-window in ms (default 500)
    bin_ms        : bin size in ms (default 1)

    Returns
    -------
    acg  : (n_bins,) float64 — conditional firing rate [Hz]
    t_ms : (n_bins,) float64 — lag axis in ms
    """
    bin_s  = bin_ms / 1000.0
    half   = int(lag_ms / bin_ms)
    n_bins = 2 * half + 1
    t_ms   = np.arange(-half, half + 1, dtype=np.float64) * bin_ms

    if len(spike_times_s) < 2:
        return np.zeros(n_bins), t_ms

    st     = np.sort(np.asarray(spike_times_s, dtype=np.float64))
    counts = _acg_1d_engine(st, half, bin_s)

    acg       = counts.astype(np.float64) / (len(st) * bin_s)
    acg[half] = 0.0   # zero 0-lag bin (self-coincidence artefact)
    return acg, t_ms


def compute_acg_3d(
    spike_times_s: np.ndarray,
    lag_ms:        float = 2000.0,
    bin_ms:        float = 1.0,
    n_fr_bins:     int   = 10,
    n_log_bins:    int   = 100,
    log_start_ms:  float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    3-D autocorrelogram: log-lag × firing-rate-quantile heatmap.

    Matches the npyx / C4 classifier approach:
    (1) Compute linear-lag ACG conditioned on instantaneous firing rate bin.
    (2) Re-sample lag axis to log-spaced bins.

    Parameters
    ----------
    spike_times_s : spike times in seconds (any order)
    lag_ms        : half-window in ms for the linear computation (default 2000)
    bin_ms        : linear bin size in ms (default 1)
    n_fr_bins     : number of firing-rate quantile bins (default 10)
    n_log_bins    : log-spaced lag bins for the positive half (default 100)
    log_start_ms  : first log bin position in ms (default 1.0)

    Returns
    -------
    acg_3d   : (2*n_log_bins+1, n_fr_bins) float64 — conditional rate [Hz]
    t_log    : (2*n_log_bins+1,) float64 — log-spaced lag axis [ms]
    fr_edges : (n_fr_bins+1,) float64 — FR quantile bin edges [Hz]
    """
    bin_s      = bin_ms / 1000.0
    half       = int(lag_ms / bin_ms)
    n_lag      = 2 * half + 1
    empty_3d   = np.zeros((2 * n_log_bins + 1, n_fr_bins))
    t_log_dummy = np.zeros(2 * n_log_bins + 1)

    if len(spike_times_s) < 2:
        return empty_3d, t_log_dummy, np.zeros(n_fr_bins + 1)

    st = np.sort(np.asarray(spike_times_s, dtype=np.float64))

    # Instantaneous firing rate per spike: FR_i = 1 / ISI_{i-1}
    # Duplicate timestamps (ISI=0) produce inf; treat them as the last FR bin.
    isis          = np.diff(st)                   # (n-1,)
    with np.errstate(divide="ignore"):
        fr_values = 1.0 / isis                    # Hz  (may contain inf for ISI=0)
    fr_per_spike  = np.empty(len(st), dtype=np.float64)
    fr_per_spike[0]  = np.nan                     # no preceding ISI for first spike
    fr_per_spike[1:] = fr_values

    # FR quantile bin edges — exclude inf/nan for percentile computation
    fr_finite = fr_values[np.isfinite(fr_values)]
    if len(fr_finite) < 2:
        return empty_3d, t_log_dummy, np.zeros(n_fr_bins + 1)
    fr_edges = np.percentile(fr_finite, np.linspace(0, 100, n_fr_bins + 1))

    counts, ref_count = _acg_3d_engine(
        st, fr_per_spike, fr_edges, half, bin_s,
    )

    # Normalise each FR bin: count / (ref_count * bin_size) → Hz
    acg_linear = np.zeros((n_lag, n_fr_bins), dtype=np.float64)
    for b in range(n_fr_bins):
        if ref_count[b] > 0:
            acg_linear[:, b] = counts[:, b] / (ref_count[b] * bin_s)
    acg_linear[half, :] = 0.0   # zero 0-lag

    # Convert lag axis to log-spaced bins
    acg_3d, t_log = _convert_lag_to_log(acg_linear, bin_ms, n_log_bins, log_start_ms)

    return acg_3d, t_log, fr_edges


def build_acg_features(
    unit_ids:       np.ndarray,
    spike_times:    np.ndarray,    # samples (int64)
    spike_clusters: np.ndarray,
    sample_rate:    float = 30_000.0,
    lag_ms:         float = 2000.0,
    bin_ms:         float = 1.0,
    n_fr_bins:      int   = 10,
    n_log_bins:     int   = 100,
    log_start_ms:   float = 1.0,
    verbose:        bool  = True,
) -> dict:
    """
    Batch-compute 1-D and 3-D ACGs for a list of units.

    Parameters
    ----------
    unit_ids       : cluster IDs to process
    spike_times    : all spike times in **samples** (from spike_times.npy)
    spike_clusters : cluster label per spike (from spike_clusters.npy)
    sample_rate    : Hz (default 30 000)
    lag_ms         : ACG half-window in ms for linear computation
    bin_ms         : linear bin size in ms
    n_fr_bins      : firing-rate quantile bins for 3-D ACG
    n_log_bins     : log-spaced lag bins (positive half) for 3-D ACG
    log_start_ms   : first log bin in ms
    verbose        : print per-unit progress

    Returns
    -------
    dict
        unit_ids  : (n_units,) int64
        acg_1d    : (n_units, 2*half+1) float64   — linear lag, [Hz]
        acg_3d    : (n_units, 2*n_log_bins+1, n_fr_bins) float64
        t_ms      : (2*half+1,) linear lag axis [ms]   for acg_1d
        t_log     : (2*n_log_bins+1,) log lag axis [ms] for acg_3d
        fr_edges  : (n_units, n_fr_bins+1) float64 — per-unit FR quantile edges
    """
    unit_ids = np.asarray(unit_ids, dtype=np.int64)
    n_units  = len(unit_ids)
    half_1d  = int(lag_ms / bin_ms)
    n_lag_1d = 2 * half_1d + 1
    n_lag_3d = 2 * n_log_bins + 1

    acg_1d_all  = np.zeros((n_units, n_lag_1d),           dtype=np.float64)
    acg_3d_all  = np.zeros((n_units, n_lag_3d, n_fr_bins), dtype=np.float64)
    fr_edges_all = np.zeros((n_units, n_fr_bins + 1),      dtype=np.float64)
    t_ms_out = t_log_out = None

    if verbose:
        print(f"Computing ACGs for {n_units} units  "
              f"(lag ±{lag_ms:.0f} ms, bin {bin_ms} ms, "
              f"{n_fr_bins} FR bins, {n_log_bins} log-lag bins)\n"
              "Note: first call compiles Numba JIT (~1-2 s) ...")

    pad = len(str(n_units))
    for i, uid in enumerate(unit_ids):
        st_samples = spike_times[spike_clusters == uid]
        st_s       = st_samples.astype(np.float64) / sample_rate

        acg_1d_all[i], t_ms_out = compute_acg(
            st_s, lag_ms=lag_ms, bin_ms=bin_ms,
        )
        acg_3d_all[i], t_log_out, fr_edges_all[i] = compute_acg_3d(
            st_s, lag_ms=lag_ms, bin_ms=bin_ms,
            n_fr_bins=n_fr_bins, n_log_bins=n_log_bins,
            log_start_ms=log_start_ms,
        )

        if verbose:
            duration = st_s[-1] - st_s[0] if len(st_s) > 1 else 0.0
            fr       = len(st_s) / duration if duration > 0 else 0.0
            print(f"  [{i+1:>{pad}}/{n_units}] unit {int(uid):5d}  "
                  f"{len(st_s):6,} spikes  FR={fr:.1f} Hz")

    return {
        "unit_ids":  unit_ids,
        "acg_1d":    acg_1d_all,
        "acg_3d":    acg_3d_all,
        "t_ms":      t_ms_out,
        "t_log":     t_log_out,
        "fr_edges":  fr_edges_all,
    }
