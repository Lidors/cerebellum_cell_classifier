# CLAUDE.md — Cerebellum Cell Classifier

This file gives Claude Code the context it needs to work on this project effectively.

## What This Project Does

Extracts spike waveform and autocorrelogram (ACG) features from Neuropixels recordings sorted with Kilosort 4, and displays them in an interactive PyQt5 GUI. Based on the approach of Herzfeld et al. 2025 (eLife). The end goal is a deep-learning classifier for cerebellar cell types (PC, CF, MLI, GC, UBC, MF, GoC).

## Architecture

```
cerebellum_cell_classifier/
├── run_extraction.py        ← main pipeline (CLI + Python API)
├── viewer.py                ← GUI launcher
│
├── io/
│   └── kilosort.py          ← loads KS4 output (.npy files + cluster_info.tsv)
│
├── features/
│   ├── waveform.py          ← mean waveform extraction from raw .ap.bin
│   ├── acg.py               ← 1D and 3D autocorrelogram (Numba JIT inner loops)
│   └── ccg.py               ← cross-correlogram, auto-labels PC/CF/MLI pairs
│
├── gui/
│   ├── main.py              ← QApplication entry point, dark theme
│   ├── app_window.py        ← main window, session tabs, keyboard shortcuts
│   ├── data_store.py        ← wraps .npz features file for GUI access
│   ├── unit_table.py        ← sortable/filterable unit table widget
│   ├── plots_panel.py       ← waveform + ACG 1D/3D plots (pyqtgraph)
│   ├── controls.py          ← right-side control panel
│   └── pair_panel.py        ← CCG pair explorer
│
├── notebooks/
│   ├── run_extraction_batch.ipynb   ← batch extraction for multiple sessions
│   └── test_waveforms.ipynb         ← testing / exploration notebook
│
└── docs/
    └── features.md          ← technical reference for all feature definitions
```

## Key Data Flow

1. `io/kilosort.py` → loads spike times, cluster IDs, channel map from KS4 folder
2. `features/waveform.py` → reads raw binary, extracts mean waveforms (N × 8ch × 81 samples)
3. `features/acg.py` → computes 1D ACG (N × 4001 bins) and 3D ACG (N × 201 × 10)
4. `features/ccg.py` → optional CCG-based auto-labeling of PC/CF/MLI pairs
5. Output → `{session}_features.npz` + `{session}_table.csv`
6. `viewer.py` → loads `.npz`, interactive exploration in GUI

## Environment Setup

```bash
conda create -n cerebellum_clf python=3.10 -y
conda activate cerebellum_clf
pip install numpy scipy pandas numba pyqt5 pyqtgraph matplotlib scikit-learn tqdm
```

## Common Commands

```bash
# Run feature extraction
python run_extraction.py --session "path/to/ks4_output" --output "path/to/save"

# Launch viewer
python viewer.py

# Launch viewer with a specific file
python viewer.py "path/to/session_features.npz"
```

## Output Format

The `.npz` output contains these arrays:

| Array | Shape | Description |
|-------|-------|-------------|
| `unit_ids` | `(N,)` | Cluster IDs |
| `labels` | `(N,)` | Cell-type strings |
| `mean_waveforms` | `(N, 8, 81)` | Mean waveform, 8 ch × 81 samples |
| `std_waveforms` | `(N, 8, 81)` | Waveform SD |
| `acg_1d` | `(N, 4001)` | 1D ACG, normalized to Hz |
| `acg_3d` | `(N, 201, 10)` | 3D ACG, log-lag × FR-quantile |
| `t_ms` | `(4001,)` | Lag axis for 1D ACG (ms) |
| `t_log` | `(201,)` | Lag axis for 3D ACG (ms, log-spaced) |

## Development Notes

- **Numba JIT:** `features/acg.py` and `features/ccg.py` use Numba for the inner loops. First run is slow (~10s for compilation), subsequent runs use cached compiled code.
- **Performance-critical paths:** waveform extraction reads raw binary in chunks — avoid loading the full `.ap.bin` into memory.
- **GUI style:** dark theme via custom QPalette, all plots use pyqtgraph (not matplotlib). Colors: BG `#1e1e2e`, FG `#e0e0e0`.
- **No global state:** each session in the GUI gets its own `SessionData` object.
- **Phase 4 (classifier) is not yet implemented** — `torch` is in `requirements.txt` as a placeholder.

## What NOT to Commit

- `test_output/` — generated feature files (.npz, .csv)
- `debug_*.py` / `debug_*.png` — personal debugging scripts
- `__pycache__/` — compiled Python and Numba cache
- Any `.npy`, `.npz`, `.bin`, `.mat` files — these are data, not code

## Citation

Herzfeld D.J. et al. (2025). *Mapping cerebellar cell-type diversity using deep learning on extracellular recordings.* eLife.
