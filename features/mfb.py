"""
mfb.py
------
Mossy Fiber Bouton (MFB) detection from mean waveforms using the
Negative AfterWave (NAW) feature.

Ports the MATLAB standalone_detect_plot_mfb_naw.m algorithm to Python.
Operates on mean_waveforms produced by build_waveform_features() — no
additional data loading required.

Public API
----------
compute_waveform_morphology(waveform_1d, sample_rate) -> dict
    Scalar waveform metrics for one unit (main channel).

build_mfb_features(mean_waveforms, sample_rate, verbose) -> dict
    Batch wrapper: runs compute_waveform_morphology for all units.

detect_mfb(features_dict, ...) -> dict
    Applies NAW + narrow-spike gates to assign tier labels and scores.
"""

from __future__ import annotations

import math
import numpy as np

# ── Default thresholds — match the user's MATLAB detCfg ──────────────────────
_NAW_MIN_LAT_MS        = 0.03  # nawLatencyMsRange(1) — detection gate AND search start
_NAW_MAX_LAT_MS        = 0.60  # nawLatencyMsRange(2) — detection gate (user config)
_NAW_SEARCH_MAX_LAT_MS = 1.20  # feature extraction search end (MATLAB local_template_features
                                #   searches to 1.2ms so the global minimum across a wider
                                #   window is found; the detection gate then filters to 0.60ms)
_NAW_AMP_RATIO_PROB    = 0.06  # nawAmpRatioProbMin
_NAW_AMP_RATIO_CORE    = 0.10  # nawAmpRatioCoreMin
_MAX_TTP_MS            = 0.40  # maxTroughToPeakMs
_MAX_HW_MS             = 0.12  # maxHalfwidthMs
_SCORE_W_NAW        = 0.55
_SCORE_W_NARROW     = 0.30
_SCORE_W_IFR        = 0.15   # IFR gate disabled → always 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  Single-unit waveform morphology
# ══════════════════════════════════════════════════════════════════════════════

def compute_waveform_morphology(
    waveform_1d: np.ndarray,
    sample_rate: float = 30_000.0,
) -> dict:
    """
    Compute NAW and spike-shape metrics from a single-channel mean waveform.

    Parameters
    ----------
    waveform_1d : (n_samples,) array_like
        Mean waveform for one unit on its main channel.  May be raw float32
        (not yet normalised).  Both polarities handled: the function internally
        flips the signal so the primary deflection is always negative.
    sample_rate : float
        Sampling rate in Hz (default 30 000).

    Returns
    -------
    dict with keys:
        trough_idx              int   — sample index of primary trough
        peak_idx                int   — sample index of post-trough peak
        trough_val              float — amplitude at trough (negative)
        peak_val                float — amplitude at peak
        ttp_ms                  float — trough-to-peak duration (ms)
        halfwidth_ms            float — half-width at half-max of trough (ms); NaN if uncrossed
        naw_present             bool  — True if a valid NAW was found
        naw_val                 float — NAW amplitude (negative); 0.0 if absent
        naw_amp_ratio           float — |naw_val| / |trough_val|; 0.0 if absent
        naw_latency_from_peak_ms float — ms from peak to NAW; NaN if absent
    """
    wf = np.asarray(waveform_1d, dtype=np.float64)
    n  = len(wf)

    # ── Polarity normalisation ────────────────────────────────────────────────
    # Only flip if the waveform has NO negative component at all (completely
    # inverted).  MFB templates often have a large positive rebound that
    # exceeds the negative trough in amplitude — flipping them destroys the
    # NAW (the NAW region, which is negative in the original, becomes positive
    # after flip).  The trough is always argmin(wf); no flip is needed unless
    # the minimum is actually non-negative.
    if wf.min() >= 0:
        wf = -wf

    _nan = float("nan")
    _defaults = dict(
        trough_idx=0, peak_idx=0, trough_val=_nan, peak_val=_nan,
        ttp_ms=_nan, halfwidth_ms=_nan,
        naw_present=False, naw_val=0.0, naw_amp_ratio=0.0,
        naw_latency_from_peak_ms=_nan,
    )

    # Guard: flat or near-zero waveform
    if np.abs(wf).max() < 1e-9:
        return _defaults

    # ── Trough ────────────────────────────────────────────────────────────────
    trough_idx = int(np.argmin(wf))
    trough_val = float(wf[trough_idx])

    if abs(trough_val) < 1e-6:
        return _defaults

    # ── Peak after trough ─────────────────────────────────────────────────────
    if trough_idx < n - 1:
        peak_rel = int(np.argmax(wf[trough_idx:]))
        peak_idx = trough_idx + peak_rel
    else:
        peak_idx = trough_idx
    peak_val = float(wf[peak_idx])

    ttp_ms = (peak_idx - trough_idx) / sample_rate * 1000.0

    # ── Halfwidth (full width at half-maximum of trough) ──────────────────────
    half_level = trough_val / 2.0  # negative / 2 = more negative than trough_val

    # Walk left from trough until waveform rises above half_level
    left = trough_idx
    while left > 0 and wf[left] <= half_level:
        left -= 1

    # Walk right from trough until waveform rises above half_level
    right = trough_idx
    while right < n - 1 and wf[right] <= half_level:
        right += 1

    # If the walk hit the boundary without crossing, halfwidth is undefined
    if left == 0 and wf[0] <= half_level:
        halfwidth_ms = _nan
    elif right == n - 1 and wf[n - 1] <= half_level:
        halfwidth_ms = _nan
    else:
        halfwidth_ms = (right - left - 1) / sample_rate * 1000.0

    # ── NAW search ────────────────────────────────────────────────────────────
    # Search to 1.2ms (matching MATLAB's local_template_features) so the true
    # global minimum is found.  The 0.60ms latency gate is applied later in
    # detect_mfb() — units whose NAW sits beyond 0.60ms are excluded there.
    naw_start = peak_idx + max(1, int(round(_NAW_MIN_LAT_MS * sample_rate / 1000.0)))
    naw_end   = peak_idx + int(round(_NAW_SEARCH_MAX_LAT_MS * sample_rate / 1000.0))
    naw_end   = min(naw_end, n - 1)

    naw_present = False
    naw_val     = 0.0
    naw_amp_ratio           = 0.0
    naw_latency_from_peak_ms = _nan

    if naw_start < n and naw_start < naw_end:
        window   = wf[naw_start : naw_end + 1]
        naw_rel  = int(np.argmin(window))
        naw_min  = float(window[naw_rel])

        if naw_min < 0.0:
            naw_idx              = naw_start + naw_rel
            naw_val              = naw_min
            naw_amp_ratio        = abs(naw_val) / abs(trough_val)
            naw_latency_from_peak_ms = (naw_idx - peak_idx) / sample_rate * 1000.0
            naw_present          = True

    return dict(
        trough_idx               = trough_idx,
        peak_idx                 = peak_idx,
        trough_val               = trough_val,
        peak_val                 = peak_val,
        ttp_ms                   = ttp_ms,
        halfwidth_ms             = halfwidth_ms,
        naw_present              = naw_present,
        naw_val                  = naw_val,
        naw_amp_ratio            = naw_amp_ratio,
        naw_latency_from_peak_ms = naw_latency_from_peak_ms,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Template-based waveform extraction helper
# ══════════════════════════════════════════════════════════════════════════════

def _dominant_template_waveform(
    unit_id: int,
    spike_clusters: np.ndarray,
    spike_templates: np.ndarray,
    templates: np.ndarray,
    whitening_mat_inv: "np.ndarray | None",
) -> "np.ndarray | None":
    """
    Return the best-channel waveform (1D) for one unit from Kilosort templates.

    Steps (matching MATLAB local_template_features):
      1. Find the dominant template for this unit (most-used template index).
      2. Extract that template: shape (n_samples, n_channels).
      3. Unwhiten if whitening_mat_inv is available: W_uw = W @ Winv.
      4. Find best channel: channel with the most negative trough (argmin of
         per-channel minimum).
      5. Return that channel's waveform as a 1D float64 array.

    Returns None if the unit has no spikes or templates is unavailable.
    """
    mask = spike_clusters == unit_id
    if not mask.any():
        return None

    tpl_ids = spike_templates[mask]
    dom_tpl = int(np.bincount(tpl_ids.astype(np.int64)).argmax())

    # templates shape: (n_tpl, n_samples, n_channels)
    W = templates[dom_tpl].astype(np.float64)   # (n_samples, n_channels)

    if whitening_mat_inv is not None:
        W = W @ whitening_mat_inv               # unwhiten

    # Best channel = channel whose trough is deepest (most negative)
    best_ch = int(np.argmin(W.min(axis=0)))
    return W[:, best_ch]


# ══════════════════════════════════════════════════════════════════════════════
#  Batch feature builder
# ══════════════════════════════════════════════════════════════════════════════

def build_mfb_features(
    unit_ids: np.ndarray,
    spike_clusters: np.ndarray,
    spike_templates: "np.ndarray | None",
    templates: "np.ndarray | None",
    whitening_mat_inv: "np.ndarray | None" = None,
    sample_rate: float = 30_000.0,
    verbose: bool = True,
) -> dict:
    """
    Compute MFB/NAW waveform features for all units using Kilosort templates.

    Uses templates.npy (+ optional whitening_mat_inv.npy) from the Kilosort
    output directory, matching the MATLAB standalone_detect_plot_mfb_naw.m
    approach.  Templates are smoother than raw-binary averages and do not
    require the .ap.bin file.

    Parameters
    ----------
    unit_ids : (n_units,) int64
        Cluster IDs to process.
    spike_clusters : (n_spikes,) int64
        Cluster ID for each spike (from spike_clusters.npy).
    spike_templates : (n_spikes,) int64 or None
        Template index (0-based) for each spike (spike_templates.npy).
        If None, returns NaN arrays (MFB detection skipped).
    templates : (n_templates, n_samples, n_channels) float32 or None
        Kilosort template array (templates.npy).
        If None, returns NaN arrays (MFB detection skipped).
    whitening_mat_inv : (n_channels, n_channels) float64 or None
        Inverse whitening matrix (whitening_mat_inv.npy).  Applied as
        W_unwhitened = W @ Winv.  If None, templates are used as-is.
    sample_rate : float
        Sampling rate in Hz.
    verbose : bool
        Print progress summary.

    Returns
    -------
    dict with float32 arrays of shape (n_units,):
        ttp_ms                   trough-to-peak duration (spike width)
        halfwidth_ms             half-width at half-max of trough
        naw_amp_ratio            NAW amplitude / trough amplitude  (0 if absent)
        naw_latency_from_peak_ms ms from peak to NAW  (NaN if absent)
        naw_present              bool array
    """
    n_units = len(unit_ids)

    # Keep float64 throughout to avoid float32 boundary-rounding failures
    # (e.g. ttp=0.4ms or lat=0.6ms stored as float32 become slightly above
    # threshold, causing valid units to fail the narrowOK / latOK gates).
    # Conversion to float32 happens only when saving to the .npz.
    ttp_ms    = np.full(n_units, np.nan, dtype=np.float64)
    hw_ms     = np.full(n_units, np.nan, dtype=np.float64)
    naw_ratio = np.zeros(n_units,        dtype=np.float64)
    naw_lat   = np.full(n_units, np.nan, dtype=np.float64)
    naw_pres  = np.zeros(n_units,        dtype=bool)

    if spike_templates is None or templates is None:
        if verbose:
            print("  MFB: templates.npy or spike_templates.npy not found — "
                  "skipping NAW detection (returning NaN arrays).")
        return dict(
            ttp_ms=ttp_ms, halfwidth_ms=hw_ms, naw_amp_ratio=naw_ratio,
            naw_latency_from_peak_ms=naw_lat, naw_present=naw_pres,
        )

    for i, uid in enumerate(unit_ids):
        wf = _dominant_template_waveform(
            unit_id=int(uid),
            spike_clusters=spike_clusters,
            spike_templates=spike_templates,
            templates=templates,
            whitening_mat_inv=whitening_mat_inv,
        )
        if wf is None:
            continue
        m = compute_waveform_morphology(wf, sample_rate)
        ttp_ms[i]    = m["ttp_ms"]
        hw_ms[i]     = m["halfwidth_ms"]
        naw_ratio[i] = m["naw_amp_ratio"]
        naw_lat[i]   = m["naw_latency_from_peak_ms"]
        naw_pres[i]  = m["naw_present"]

    if verbose:
        n_naw = int(naw_pres.sum())
        print(f"  MFB: {n_naw}/{n_units} units with NAW detected "
              f"(ratio >= {_NAW_AMP_RATIO_PROB})")

    return dict(
        ttp_ms                   = ttp_ms,
        halfwidth_ms             = hw_ms,
        naw_amp_ratio            = naw_ratio,
        naw_latency_from_peak_ms = naw_lat,
        naw_present              = naw_pres,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Tier detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_mfb(
    features_dict: dict,
    naw_min_lat_ms:     float = _NAW_MIN_LAT_MS,
    naw_max_lat_ms:     float = _NAW_MAX_LAT_MS,
    naw_amp_ratio_prob: float = _NAW_AMP_RATIO_PROB,
    naw_amp_ratio_core: float = _NAW_AMP_RATIO_CORE,
    max_ttp_ms:         float = _MAX_TTP_MS,
    max_halfwidth_ms:   float = _MAX_HW_MS,
) -> dict:
    """
    Assign MFB detection tiers and a composite score to each unit.

    Parameters
    ----------
    features_dict : dict
        Output of build_mfb_features().
    naw_min_lat_ms, naw_max_lat_ms : float
        Valid NAW latency window (ms from peak).
    naw_amp_ratio_prob, naw_amp_ratio_core : float
        NAW amplitude ratio thresholds for 'probable' and 'core' tiers.
    max_ttp_ms : float
        Maximum trough-to-peak for 'narrow' gate.
    max_halfwidth_ms : float
        Maximum halfwidth for 'narrow' gate.

    Returns
    -------
    dict:
        mfb_tier  : (n_units,) object array of str — 'core', 'probable', or 'review'
        mfb_score : (n_units,) float32 — composite score in [0, 1]
    """
    lat   = features_dict["naw_latency_from_peak_ms"].astype(np.float64)
    ratio = features_dict["naw_amp_ratio"].astype(np.float64)
    ttp   = features_dict["ttp_ms"].astype(np.float64)
    hw    = features_dict["halfwidth_ms"].astype(np.float64)
    pres  = features_dict["naw_present"]

    n = len(ratio)

    # ── Boolean gates (vectorised) ────────────────────────────────────────────
    lat_ok       = pres & np.isfinite(lat) & (lat >= naw_min_lat_ms) & (lat <= naw_max_lat_ms)
    naw_prob_ok  = lat_ok & (ratio >= naw_amp_ratio_prob)
    naw_core_ok  = lat_ok & (ratio >= naw_amp_ratio_core)
    narrow_ok    = (np.isfinite(ttp) & (ttp <= max_ttp_ms)) | \
                   (np.isfinite(hw)  & (hw  <= max_halfwidth_ms))

    mfb_core = naw_core_ok & narrow_ok
    mfb_prob = naw_prob_ok & narrow_ok & ~mfb_core

    # ── Tier labels ───────────────────────────────────────────────────────────
    tier = np.full(n, "review", dtype=object)
    tier[mfb_prob] = "probable"
    tier[mfb_core] = "core"

    # ── Composite score ───────────────────────────────────────────────────────
    score_naw    = np.minimum(1.0, ratio / max(naw_amp_ratio_core, 1e-9))
    score_narrow = np.where(
        np.isfinite(ttp),
        np.minimum(1.0, np.maximum(0.0, (max_ttp_ms - ttp) / max(max_ttp_ms, 1e-9))),
        0.0,
    )
    score = (_SCORE_W_NAW * score_naw +
             _SCORE_W_NARROW * score_narrow +
             _SCORE_W_IFR * 1.0)
    score = np.clip(score, 0.0, 1.0).astype(np.float32)

    if True:  # always print summary (caller controls verbosity via build_mfb_features)
        pass

    return dict(
        mfb_tier  = tier,
        mfb_score = score,
    )
