"""
classifier.py
-------------
Load a trained model folder and run cell-type inference on a SessionData object.

Backend selection (automatic)
------------------------------
  PyTorch available  -> use .pt VAE checkpoints (faster, GPU-capable)
  PyTorch missing    -> use .onnx encoder files via onnxruntime (lightweight, CPU-only)

Install for CPU-only machines:
  pip install onnxruntime xgboost scikit-learn numpy

Model folder layout (produced by train_final_classifier.py)
-----------------------------------------------------------
  wf_vae.pt          -- WFConvVAE state_dict          (PyTorch path)
  acg_vae.pt         -- ACGConvVAE state_dict          (PyTorch path)
  wf_encoder.onnx    -- WF encoder, returns mu only    (ONNX path)
  acg_encoder.onnx   -- ACG encoder, returns mu only   (ONNX path)
  xgb_clf.json       -- XGBoost model (native JSON)
  scaler.pkl         -- sklearn StandardScaler
  label_encoder.pkl  -- sklearn LabelEncoder
  clf_meta.json      -- {classes, feature_cols, n_chan_use, latent_dim, layer_cats}

Public API
----------
  load_model_folder(folder) -> dict
  run_inference(session_data, models) -> (labels: ndarray[str], conf: ndarray[float])
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from cerebellum_cell_classifier.autoencoders.transforms import (
    normalize_waveforms,
    normalize_acg3d,
)

# ── Optional backends ─────────────────────────────────────────────────────────

try:
    import torch
    from cerebellum_cell_classifier.autoencoders.models import WFConvVAE, ACGConvVAE
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ── Folder loading ────────────────────────────────────────────────────────────

def load_model_folder(folder: str | Path) -> dict:
    """
    Load all components from a model folder.

    Automatically selects backend:
      - PyTorch if available (preferred)
      - onnxruntime otherwise

    Returns a dict with keys:
      backend      : "torch" or "onnx"
      wf_vae       : WFConvVAE          (torch backend only)
      acg_vae      : ACGConvVAE         (torch backend only)
      wf_sess      : ort.InferenceSession  (onnx backend only)
      acg_sess     : ort.InferenceSession  (onnx backend only)
      clf          : XGBClassifier
      scaler       : StandardScaler
      le           : LabelEncoder
      meta         : dict
    """
    folder = Path(folder)

    if not HAS_XGB:
        raise ImportError(
            "xgboost is required. Install with: pip install xgboost"
        )

    for name in ("clf_meta.json", "xgb_clf.json", "scaler.pkl", "label_encoder.pkl"):
        _check(folder / name, name)

    with open(folder / "clf_meta.json") as f:
        meta = json.load(f)

    # XGBoost
    clf = xgb.XGBClassifier()
    clf.load_model(str(folder / "xgb_clf.json"))

    # Scaler + LabelEncoder
    with open(folder / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(folder / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    n_chan_use = int(meta["n_chan_use"])
    latent_dim = int(meta["latent_dim"])

    result = dict(clf=clf, scaler=scaler, le=le, meta=meta)

    if HAS_TORCH:
        # ── PyTorch backend ───────────────────────────────────────────────────
        for name in ("wf_vae.pt", "acg_vae.pt"):
            _check(folder / name, name)

        wf_vae = WFConvVAE(n_channels=n_chan_use, n_timepoints=81, latent_dim=latent_dim)
        wf_vae.load_state_dict(
            torch.load(folder / "wf_vae.pt", map_location="cpu", weights_only=True)
        )
        wf_vae.eval()

        acg_vae = ACGConvVAE(latent_dim=latent_dim)
        acg_vae.load_state_dict(
            torch.load(folder / "acg_vae.pt", map_location="cpu", weights_only=True)
        )
        acg_vae.eval()

        result.update(backend="torch", wf_vae=wf_vae, acg_vae=acg_vae)

    elif HAS_ORT:
        # ── ONNX Runtime backend ──────────────────────────────────────────────
        for name in ("wf_encoder.onnx", "acg_encoder.onnx"):
            _check(folder / name, name)

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3   # suppress onnxruntime info spam
        wf_sess  = ort.InferenceSession(
            str(folder / "wf_encoder.onnx"),  sess_opts, providers=["CPUExecutionProvider"]
        )
        acg_sess = ort.InferenceSession(
            str(folder / "acg_encoder.onnx"), sess_opts, providers=["CPUExecutionProvider"]
        )
        result.update(backend="onnx", wf_sess=wf_sess, acg_sess=acg_sess)

    else:
        raise ImportError(
            "No inference backend found.\n"
            "Install PyTorch: https://pytorch.org\n"
            "  OR\n"
            "Install onnxruntime (CPU-only, lightweight): pip install onnxruntime"
        )

    return result


def _check(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Model folder is missing required file: {name}")


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(session_data, models: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the classifier on all units in *session_data*.

    Returns
    -------
    labels : (N,) str ndarray — predicted class per unit
    conf   : (N,) float32 ndarray — probability of the predicted class
    """
    meta       = models["meta"]
    clf        = models["clf"]
    scaler     = models["scaler"]
    le         = models["le"]
    n_chan_use = int(meta["n_chan_use"])
    layer_cats = meta["layer_cats"]
    n          = session_data.n_units

    # ── 1. Normalise waveforms & ACGs ─────────────────────────────────────────
    wf_raw  = session_data.mean_waveforms[:, :n_chan_use, :]   # (N, n_chan, 81)
    wf_norm = normalize_waveforms(wf_raw)                       # (N, n_chan, 81)

    acg_raw  = session_data.acg_3d                             # (N, 201, 10)
    acg_norm = normalize_acg3d(acg_raw)                        # (N, 1, 10, 101)

    # ── 2. Encode ─────────────────────────────────────────────────────────────
    if models["backend"] == "torch":
        wf_t  = torch.tensor(wf_norm[:, np.newaxis], dtype=torch.float32)
        acg_t = torch.tensor(acg_norm,               dtype=torch.float32)
        with torch.no_grad():
            wf_z  = models["wf_vae"].encode(wf_t).numpy()   # (N, latent_dim)
            acg_z = models["acg_vae"].encode(acg_t).numpy() # (N, latent_dim)
    else:
        # ONNX Runtime
        wf_input  = wf_norm[:, np.newaxis].astype(np.float32)  # (N, 1, n_chan, 81)
        acg_input = acg_norm.astype(np.float32)                 # (N, 1, 10, 101)
        wf_z  = models["wf_sess"].run(None,  {"wf":  wf_input})[0]   # (N, latent_dim)
        acg_z = models["acg_sess"].run(None, {"acg": acg_input})[0]  # (N, latent_dim)

    # ── 3. Build feature matrix ───────────────────────────────────────────────
    fr_arr = np.array([session_data.get_mean_fr(i) for i in range(n)], dtype=np.float32)
    fr_arr = np.nan_to_num(fr_arr, nan=0.0)

    layer_arr = np.zeros((n, len(layer_cats)), dtype=np.float32)
    unknown_idx = layer_cats.index("unknown")
    for i in range(n):
        lyr = session_data.get_layer(i) or "unknown"
        idx = layer_cats.index(lyr) if lyr in layer_cats else unknown_idx
        layer_arr[i, idx] = 1.0

    X = np.concatenate([wf_z, acg_z, fr_arr[:, np.newaxis], layer_arr], axis=1)
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)

    # ── 4. Scale + predict ────────────────────────────────────────────────────
    X_scaled    = scaler.transform(X)
    y_enc       = clf.predict(X_scaled)
    labels_pred = le.inverse_transform(y_enc)
    proba       = clf.predict_proba(X_scaled)
    conf        = proba.max(axis=1).astype(np.float32)

    return np.array(labels_pred, dtype=object), conf
