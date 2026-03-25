"""
classifier.py
-------------
Load a trained model folder and run cell-type inference on a SessionData object.

Model folder layout (produced by train_final_classifier.py)
-----------------------------------------------------------
  wf_vae.pt       -- WFConvVAE state_dict
  acg_vae.pt      -- ACGConvVAE state_dict
  xgb_clf.json    -- XGBoost model (native JSON format)
  scaler.pkl      -- sklearn StandardScaler
  clf_meta.json   -- {classes, feature_cols, n_chan_use, latent_dim, layer_cats}

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
import torch

from cerebellum_cell_classifier.autoencoders.models import WFConvVAE, ACGConvVAE
from cerebellum_cell_classifier.autoencoders.transforms import (
    normalize_waveforms,
    normalize_acg3d,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ── Folder loading ────────────────────────────────────────────────────────────

def load_model_folder(folder: str | Path) -> dict:
    """
    Load all components from a model folder.

    Returns a dict with keys:
      wf_vae  : WFConvVAE (eval mode, on cpu)
      acg_vae : ACGConvVAE (eval mode, on cpu)
      clf     : XGBClassifier
      scaler  : StandardScaler
      meta    : dict  {classes, feature_cols, n_chan_use, latent_dim, layer_cats}
    """
    folder = Path(folder)
    _check(folder / "clf_meta.json",     "clf_meta.json")
    _check(folder / "wf_vae.pt",         "wf_vae.pt")
    _check(folder / "acg_vae.pt",        "acg_vae.pt")
    _check(folder / "xgb_clf.json",      "xgb_clf.json")
    _check(folder / "scaler.pkl",        "scaler.pkl")
    _check(folder / "label_encoder.pkl", "label_encoder.pkl")

    if not HAS_XGB:
        raise ImportError(
            "xgboost is required to load the classifier. "
            "Install it with: pip install xgboost"
        )

    with open(folder / "clf_meta.json") as f:
        meta = json.load(f)

    n_chan_use  = int(meta["n_chan_use"])
    latent_dim  = int(meta["latent_dim"])

    # WF VAE
    wf_vae = WFConvVAE(n_channels=n_chan_use, n_timepoints=81, latent_dim=latent_dim)
    wf_vae.load_state_dict(
        torch.load(folder / "wf_vae.pt", map_location="cpu", weights_only=True)
    )
    wf_vae.eval()

    # ACG VAE
    acg_vae = ACGConvVAE(latent_dim=latent_dim)
    acg_vae.load_state_dict(
        torch.load(folder / "acg_vae.pt", map_location="cpu", weights_only=True)
    )
    acg_vae.eval()

    # XGBoost
    clf = xgb.XGBClassifier()
    clf.load_model(str(folder / "xgb_clf.json"))

    # Scaler
    with open(folder / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # LabelEncoder
    with open(folder / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    return dict(wf_vae=wf_vae, acg_vae=acg_vae, clf=clf, scaler=scaler, le=le, meta=meta)


def _check(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Model folder is missing required file: {name}")


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(session_data, models: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the classifier on all units in *session_data*.

    Parameters
    ----------
    session_data : SessionData
    models       : dict returned by load_model_folder()

    Returns
    -------
    labels : (N,) str ndarray — predicted class per unit
    conf   : (N,) float32 ndarray — probability of the predicted class
    """
    meta       = models["meta"]
    wf_vae     = models["wf_vae"]
    acg_vae    = models["acg_vae"]
    clf        = models["clf"]
    scaler     = models["scaler"]
    n_chan_use = int(meta["n_chan_use"])
    classes    = meta["classes"]          # e.g. ['CF','MF','PC','others']
    layer_cats = meta["layer_cats"]
    feature_cols = meta["feature_cols"]  # 28-dim ordered list

    n = session_data.n_units

    # ── 1. Normalise & encode waveforms ──────────────────────────────────────
    wf_raw = session_data.mean_waveforms[:, :n_chan_use, :]  # (N, n_chan, 81)
    wf_norm = normalize_waveforms(wf_raw)                    # (N, n_chan, 81)
    wf_t = torch.tensor(wf_norm[:, np.newaxis], dtype=torch.float32)  # (N,1,C,81)
    with torch.no_grad():
        wf_z = wf_vae.encode(wf_t).numpy()   # (N, latent_dim)

    # ── 2. Normalise & encode 3D ACGs ────────────────────────────────────────
    acg_raw  = session_data.acg_3d                    # (N, 201, 10)
    acg_norm = normalize_acg3d(acg_raw)               # (N, 1, 10, 101)
    acg_t = torch.tensor(acg_norm, dtype=torch.float32)
    with torch.no_grad():
        acg_z = acg_vae.encode(acg_t).numpy()         # (N, latent_dim)

    # ── 3. Build feature matrix ───────────────────────────────────────────────
    # FR per unit
    fr_arr = np.array([session_data.get_mean_fr(i) for i in range(n)], dtype=np.float32)
    fr_arr = np.nan_to_num(fr_arr, nan=0.0)

    # Layer one-hots
    layer_arr = np.zeros((n, len(layer_cats)), dtype=np.float32)
    for i in range(n):
        lyr = session_data.get_layer(i) or "unknown"
        if lyr in layer_cats:
            layer_arr[i, layer_cats.index(lyr)] = 1.0
        else:
            # default to 'unknown' column
            unknown_idx = layer_cats.index("unknown")
            layer_arr[i, unknown_idx] = 1.0

    # Concatenate in the same order as training: wf | acg | fr | layer
    X = np.concatenate([wf_z, acg_z, fr_arr[:, np.newaxis], layer_arr], axis=1)
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)

    # ── 4. Scale + predict ───────────────────────────────────────────────────
    le       = models["le"]
    X_scaled = scaler.transform(X)
    y_enc        = clf.predict(X_scaled)                        # (N,) int
    labels_pred  = le.inverse_transform(y_enc)                  # (N,) str
    proba        = clf.predict_proba(X_scaled)                  # (N, n_classes)
    conf         = proba.max(axis=1).astype(np.float32)         # (N,)

    return np.array(labels_pred, dtype=object), conf
