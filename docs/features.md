# Feature Extraction — Technical Reference

## 1. Mean Waveform (`features/waveform.py`)

### What it produces
For each unit: `mean_waveforms` **(n_units × 8 channels × 81 samples)** — the average spike shape across up to 3 000 randomly drawn spikes, extracted from 8 channels centred on the unit's main channel.

### Pipeline steps

#### 1.1 Channel selection
The main channel comes from the **`ch` column** of `cluster_info.tsv` — the channel where the Kilosort 4 template has peak amplitude. The 7 nearest channels are then selected by physical distance (µm) from `channel_positions.npy`. Main channel is always index 0.

#### 1.2 Spike subsampling
Up to `max_spikes = 3 000` spikes are drawn **randomly** (without replacement, fixed seed) from all spikes assigned to the unit. This avoids biasing toward one recording epoch.

#### 1.3 Detrending — `mydetrend` equivalent
Each extracted waveform window (81 samples × 8 channels) is passed through **`scipy.signal.detrend(type='linear')`**, which fits and subtracts a least-squares line per channel. This is the exact Python equivalent of the MATLAB `mydetrend.m` function (Stark lab):
- Step 1: subtract the **mean** of the entire window (DC removal)
- Step 2: subtract the **linear trend** (centered ramp projection)

Both steps happen in one pass with `detrend(type='linear')`.

> **Why**: raw Neuropixels data has slow drifts and channel-specific DC offsets. Without detrending each spike window, the mean waveform would have large, misleading baselines.

#### 1.4 Realignment — `realign_spikes.m` equivalent
After detrending, each spike is **re-centred** so its extremum on the main channel lands on sample 40 (the window centre). The algorithm:
1. Compute the mean waveform of the main channel across all spikes.
2. Find whether the dominant deflection is positive or negative.
3. For each spike individually, find the argmax (or argmin) within ±12 samples of the expected peak.
4. Shift by `np.roll(spike, shift, axis=1)` — identical to MATLAB `circshift`.

This corrects sub-millisecond jitter in spike timing (common with threshold crossings) and sharpens the mean waveform.

#### 1.5 Mean & SD
After alignment, `mean` and `std` are computed across spikes for each channel. `std / sqrt(n_spikes)` gives the **SEM** used in plots.

#### 1.6 Normalisation (for plotting / classifier input)
`normalize_waveforms()` scales each unit so that peak-to-trough on the main channel = 1 and the primary deflection is **negative** (paper convention).

---

## 2. Autocorrelogram — 1D ACG (`features/acg.py`)

### What it produces
`acg_1d` **(n_units × 4 001 bins)** — conditional firing rate [Hz] at each lag from −2 000 ms to +2 000 ms (1 ms bins).

### Pipeline
The inner loop (`_acg_1d_engine`, Numba JIT) is a direct Python port of **`CCGHeart.c`** (Ken Harris / Nate lab). It uses a **bidirectional sliding window**:

1. Sort spike times.
2. For each reference spike `i`, maintain `j_start` — the leftmost spike within ±`max_lag`. `j_start` is monotonically non-decreasing (O(N) total).
3. For each neighbour `j ≠ i`, compute `diff = t_j − t_i` and increment `counts[round(diff/bin) + half_bins]`.

**Normalisation** (= `'hz'` in CCG.m):
```
acg[lag_bin] = counts[lag_bin] / (n_spikes × bin_size_s)   [Hz]
```
The 0-lag bin is zeroed (self-coincidence artefact).

---

## 3. 3D ACG (`features/acg.py`)

### What it produces
`acg_3d` **(n_units × 201 log-lag bins × 10 FR bins)** — a 2D heatmap of conditional firing rate.

Matches the **NeuroPyxels / C4 classifier** approach (`corr.crosscorr_vs_firing_rate` + `corr.convert_acg_log`).

### Pipeline

#### Step 1 — Firing-rate-conditioned ACG (linear lag)
For each reference spike, compute its **instantaneous firing rate** = 1 / ISI\_preceding [Hz]. Partition all reference spikes into **10 quantile bins** (equal number of spikes per bin). For each bin, count spikes at each lag using the same Numba sliding-window engine as the 1D ACG.

Result: **(4 001 linear-lag bins × 10 FR bins)**, normalised to conditional firing rate [Hz].

> Each column answers: *"given that the neuron was firing at this rate, how likely is a spike at lag τ?"*

#### Step 2 — Log-lag conversion
The lag axis is re-sampled from **linear → log-spaced** (100 bins from 0.8 ms to 2 000 ms per side), mirrored for symmetry.

Result: **(201 log-lag bins × 10 FR bins)**.

> The log axis zooms into the fast timescale (refractory period, burst structure at 1–10 ms) while still capturing slow modulations out to 2 s.

#### Normalisation
Same as 1D: conditional rate in Hz. For classifier input, the matrix is ravelled to **2 010 features** and scaled ×10 (matching the C4 pipeline).

### Parameters (defaults match NeuroPyxels / C4)

| Parameter | Value | Notes |
|---|---|---|
| `lag_ms` | 2 000 | half-window for linear computation |
| `bin_ms` | 1 | linear bin size |
| `n_fr_bins` | 10 | firing-rate quantile bins |
| `n_log_bins` | 100 | log-lag bins per side |
| `log_start_ms` | 0.8 | first log bin (ms) |
