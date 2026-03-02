# Cerebellum Cell Classifier

Extract waveform and autocorrelogram (ACG) features from Neuropixels recordings sorted with **Kilosort 4**, and inspect them in an interactive GUI.

Based on the approach of *Herzfeld et al. 2025 (eLife)*.

---

## Requirements

- **OS:** Windows 10/11 (Linux/macOS should work but are untested)
- **Python:** 3.10 or 3.11
- **Kilosort 4** spike-sorting output (`.npy` files + `cluster_info.tsv`)
- Raw binary file (`*ap.bin`) for waveform extraction

---

## Installation

### 1. Install Miniconda or Anaconda

Download from https://www.anaconda.com/download and follow the installer.

### 2. Clone or download the repository

```bash
git clone https://github.com/Lidors/cerebellum_cell_classifier.git
```

Or download and extract the ZIP from GitHub.

### 3. Create the conda environment

Open an **Anaconda Prompt** (or terminal), navigate to the project folder, and run:

```bash
conda create -n cerebellum_clf python=3.10 -y
conda activate cerebellum_clf
```

### 4. Install dependencies

```bash
pip install numpy scipy pandas numba pyqt5 pyqtgraph matplotlib scikit-learn tqdm
```

> **Note:** `torch` and `umap-learn` are listed in `requirements.txt` for the classifier
> (Phase 4, not yet implemented). You can skip them for now.

### 5. Verify the installation

```bash
python -c "import numpy, scipy, pandas, numba, PyQt5, pyqtgraph; print('All OK')"
```

---

## Project structure

```
cerebellum_cell_classifier/
├── run_extraction.py          ← main extraction script (CLI + Python API)
├── viewer.py                  ← GUI launcher
├── requirements.txt
│
├── io/
│   └── kilosort.py            ← load Kilosort 4 output
├── features/
│   ├── waveform.py            ← mean waveform extraction
│   └── acg.py                 ← 1D and 3D autocorrelogram
├── gui/
│   ├── main.py                ← app entry point
│   ├── app_window.py          ← main window (multi-session tabs)
│   ├── plots_panel.py         ← waveform + ACG plots
│   ├── controls.py            ← right-panel controls
│   ├── unit_table.py          ← sortable unit table
│   └── data_store.py          ← loads .npz into memory
├── notebooks/
│   └── run_extraction_batch.ipynb   ← batch extraction notebook
└── docs/
    └── features.md            ← technical details of all features
```

---

## Step 1 — Extract features

### Option A: Jupyter notebook (recommended for batch processing)

Open `notebooks/run_extraction_batch.ipynb`.
Edit the `SESSIONS` list at the top of the notebook and run all cells.

```python
SESSIONS = [
    {
        "session_path": r"E:\data\AA23\AA23_05",   # Kilosort 4 output folder
        # "bin_path":   r"E:\data\AA23\AA23_05\AA23_05_g0_tcat.imec0.ap.bin",  # optional, auto-detected
        # "output_path": r"C:\data\features",       # optional, defaults to <session>/features/
        "labels": {           # optional expert labels  {unit_id: "CellType"}
            5:   "PC",
            124: "MLI",
        },
    },
    {
        "session_path": r"E:\data\AA24\AA24_01",
    },
]
```

The notebook will print progress for each session and display a summary table when done.

### Option B: Python script

```python
from cerebellum_cell_classifier.run_extraction import run_extraction

run_extraction(
    session_path = r"E:\data\AA23\AA23_05",
    bin_path     = r"E:\data\AA23\AA23_05\AA23_05_g0_tcat.imec0.ap.bin",  # optional
    labels       = {5: "PC", 124: "MLI"},   # optional
    output_path  = r"C:\data\features",     # optional
)
```

### Option C: Command line

```bash
conda activate cerebellum_clf
python run_extraction.py ^
    --session  "E:\data\AA23\AA23_05" ^
    --bin      "E:\data\AA23\AA23_05\AA23_05_g0_tcat.imec0.ap.bin" ^
    --labels   5:PC 124:MLI ^
    --output   "C:\data\features"
```

(`--bin`, `--labels`, and `--output` are all optional.)

### What does the extraction need?

The `session_path` folder must contain the standard Kilosort 4 output:

| File | Description |
|------|-------------|
| `spike_times.npy` | Spike timestamps (samples) |
| `spike_clusters.npy` | Cluster ID per spike |
| `channel_map.npy` | Channel index mapping |
| `channel_positions.npy` | Channel x/y positions (µm) |
| `cluster_info.tsv` | Per-cluster metadata (group, depth, etc.) |
| `*ap.bin` | Raw binary file for waveform extraction |

Only **good units** (`group == "good"` in `cluster_info.tsv`) are processed.

### Output files

Both files are written to `output_path/` (default: `<session_path>/features/`):

| File | Contents |
|------|----------|
| `{session}_features.npz` | All features — pass this to the viewer |
| `{session}_table.csv` | Per-unit table (depth, FR, label, quality metrics) |

The `.npz` contains:

| Array | Shape | Description |
|-------|-------|-------------|
| `unit_ids` | `(N,)` | Cluster IDs |
| `labels` | `(N,)` | Cell-type labels (str) |
| `mean_waveforms` | `(N, 8, 81)` | Mean spike waveform, 8 channels × 81 samples |
| `std_waveforms` | `(N, 8, 81)` | Waveform standard deviation |
| `acg_1d` | `(N, 4001)` | 1D ACG, normalised to Hz |
| `acg_3d` | `(N, 201, 10)` | 3D ACG — log-lag × FR-quantile bin |
| `t_ms` | `(4001,)` | Lag axis for 1D ACG (ms) |
| `t_log` | `(201,)` | Lag axis for 3D ACG (ms, log-spaced) |

---

## Step 2 — Open the viewer

```bash
conda activate cerebellum_clf
python viewer.py
```

A file-open dialog will appear. Select the `*_features.npz` file produced in Step 1.

You can also pass the file directly:

```bash
python viewer.py "C:\data\features\AA23_05_features.npz"
```

### GUI overview

```
┌─ Session tabs ─────────────────────────────────────────────────┐
│ AA23_05 ×  AA24_01 ×  [+]                                      │
├──────────┬─────────────────────────────────┬───────────────────┤
│  Unit    │  Waveform    │  1D ACG           │  Controls         │
│  table   │              │                   │  ─────────────    │
│          ├──────────────┴───────────────────┤  Navigation       │
│  filter  │  3D ACG heatmap                  │  Waveform         │
│  sort    │  (log-lag × FR-quantile)         │  1D ACG           │
│  label   │                                  │  3D ACG           │
└──────────┴──────────────────────────────────┴───────────────────┘
```

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` | Previous / next unit |
| `Ctrl+Up` / `Ctrl+Down` | Increase / decrease waveform scale |
| `Ctrl+O` | Add a session (new tab) |
| `Ctrl+S` | Save labels to CSV |
| `Ctrl+Q` | Quit |

### Labelling units

Double-click the **Label** cell in the unit table to assign a cell type from the dropdown:

`PC · CF · MLI · GC · UBC · MF · GoC · unknown`

Press `Ctrl+S` to export all labels to a CSV file.

### 3D ACG controls

| Control | Description |
|---------|-------------|
| **Log X axis** checkbox | Toggle between log-spaced and linear ms display |
| **+/- ms** (3D ACG) | X-axis limit — zoom in/out on the lag axis |
| **Auto color** | Auto-scale colormap to 99th percentile |
| **Max** | Manual colormap ceiling |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'cerebellum_cell_classifier'`**
Make sure you are running from the folder that *contains* `cerebellum_cell_classifier/` (i.e., the parent folder), not from inside the package. Always use `python viewer.py` or `python run_extraction.py` from the project root.

**Viewer does not start / PyQt5 error**
Re-install PyQt5: `pip install --force-reinstall pyqt5`

**Numba recompilation on first run**
The first time ACG extraction runs, Numba JIT-compiles the inner loop (takes ~10 s). Subsequent runs use a cached compiled version and are fast.

**`FileNotFoundError` for `.ap.bin`**
If you have only one `.bin` file in the session folder, `bin_path` is auto-detected. If there are multiple, pass `bin_path` explicitly.

---

## Citation

If you use this code, please cite:

> Herzfeld D.J. et al. (2025). *Mapping cerebellar cell-type diversity using deep learning on extracellular recordings.* eLife.
