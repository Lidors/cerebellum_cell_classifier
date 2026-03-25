"""
train_final_classifier.py
--------------------------
Train a final XGBoost classifier on ALL ground-truth labeled data and save a
deployable model folder that can be loaded by the GUI.

Usage
-----
python train_final_classifier.py \
    --dataset-dir  Z:/loco/cell_class/datasets \
    --wf-ckpt      checkpoints/wf/wf_vae_best.pt \
    --acg-ckpt     checkpoints/acg/acg_vae_best.pt \
    --output-dir   checkpoints/final_clf \
    --n-chan-use   6

Output folder contains
----------------------
  wf_vae.pt       -- copy of WF VAE checkpoint
  acg_vae.pt      -- copy of ACG VAE checkpoint
  xgb_clf.json    -- XGBoost model (native JSON format, cross-platform)
  scaler.pkl      -- fitted sklearn StandardScaler
  clf_meta.json   -- {classes, feature_cols, n_chan_use, latent_dim, layer_cats}
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow running from any working directory
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cerebellum_cell_classifier.autoencoders.models import WFConvVAE, ACGConvVAE
from cerebellum_cell_classifier.autoencoders.transforms import (
    normalize_waveforms,
    normalize_acg3d,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ── Constants (must match train_classifier.ipynb) ────────────────────────────
CLASSES = ["CF", "MF", "PC", "others"]
LAYER_CATS = [
    "GCL", "ML", "PCL",
    "PCL_GCL_interface", "PCL_ML_interface",
    "not_cortex", "unknown",
]
LATENT_DIM = 10
WF_COLS  = [f"wf_{i}"  for i in range(LATENT_DIM)]
ACG_COLS = [f"acg_{i}" for i in range(LATENT_DIM)]
LAYER_COLS   = [f"layer_{c}" for c in LAYER_CATS]
FEATURE_COLS = WF_COLS + ACG_COLS + ["mean_fr_hz"] + LAYER_COLS


# ── Label assignment (must match notebook) ───────────────────────────────────
def assign_class(label: str, c4: str) -> str:
    label = str(label).strip()
    c4    = str(c4).strip()
    if label in ("PC", "CF", "MF"):
        return label
    if c4 in ("MLI", "GoC"):
        return "others"
    return "unknown"


# ── Encoding ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_session(npz_path: Path, wf_model: WFConvVAE, acg_model: ACGConvVAE,
                   n_chan_use: int, device: torch.device) -> pd.DataFrame:
    """Load one NPZ session, encode waveforms & ACGs, return a DataFrame."""
    data = np.load(npz_path, allow_pickle=True)
    unit_ids = data["unit_ids"]
    wf_raw   = data["mean_waveforms"]   # (N, 8, 81) float32
    acg_raw  = data["acg_3d"]           # (N, 201, 10) float64
    session  = str(data["session_name"])

    # Load companion _table.csv for FR and layer info
    csv_path = npz_path.parent / (npz_path.stem.replace("_features", "_table") + ".csv")
    if not csv_path.exists():
        csv_path = npz_path.parent / (npz_path.stem + ".csv")
    table = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    # Load manual labels from _labels.csv sidecar (preferred over NPZ labels)
    lbl_csv = npz_path.parent / (npz_path.stem.replace("_features", "_labels") + ".csv")
    manual_labels: dict[int, str] = {}
    if lbl_csv.exists():
        ldf = pd.read_csv(lbl_csv)
        if "unit_id" in ldf.columns and "label" in ldf.columns:
            manual_labels = dict(zip(ldf["unit_id"].astype(int), ldf["label"].astype(str)))

    # Normalise & encode waveforms
    wf_norm = normalize_waveforms(wf_raw[:, :n_chan_use, :])   # (N, n_chan, 81)
    wf_t = torch.tensor(wf_norm[:, None], dtype=torch.float32, device=device)  # (N,1,C,81)
    wf_z = wf_model.encode(wf_t).cpu().numpy()   # (N, latent_dim)

    # Normalise & encode ACGs
    acg_norm = normalize_acg3d(acg_raw)   # (N, 1, 10, 101)
    acg_t = torch.tensor(acg_norm, dtype=torch.float32, device=device)
    acg_z = acg_model.encode(acg_t).cpu().numpy()  # (N, latent_dim)

    rows = []
    for k, uid in enumerate(unit_ids):
        uid = int(uid)

        # Manual label (from _labels.csv) first, then NPZ labels
        label = manual_labels.get(uid, str(data["labels"][k]))
        # FR and layer from table
        fr, layer, c4 = 0.0, "unknown", ""
        if not table.empty and "unit_id" in table.columns:
            row_t = table[table["unit_id"] == uid]
            if not row_t.empty:
                fr    = float(row_t.iloc[0].get("mean_fr_hz", 0.0) or 0.0)
                layer = str(row_t.iloc[0].get("neuron_layer", "unknown") or "unknown")
                c4    = str(row_t.iloc[0].get("C4_predicted_cell_type", "") or "")

        cls = assign_class(label, c4)

        rec = {"session": session, "unit_id": uid, "class": cls}
        for j, col in enumerate(WF_COLS):
            rec[col] = float(wf_z[k, j])
        for j, col in enumerate(ACG_COLS):
            rec[col] = float(acg_z[k, j])
        rec["mean_fr_hz"] = fr
        for cat in LAYER_CATS:
            rec[f"layer_{cat}"] = float(layer == cat)
        rows.append(rec)

    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train final XGBoost classifier on all GT data")
    parser.add_argument("--dataset-dir",  required=True,  help="Folder with *_features.npz files")
    parser.add_argument("--wf-ckpt",      required=True,  help="WF VAE checkpoint (.pt)")
    parser.add_argument("--acg-ckpt",     required=True,  help="ACG VAE checkpoint (.pt)")
    parser.add_argument("--output-dir",   required=True,  help="Where to save the model folder")
    parser.add_argument("--n-chan-use",   type=int, default=6, help="# waveform channels used by WF VAE (default 6)")
    args = parser.parse_args()

    if not HAS_XGB:
        print("ERROR: xgboost is not installed. Run: pip install xgboost")
        sys.exit(1)

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    wf_ckpt     = Path(args.wf_ckpt)
    acg_ckpt    = Path(args.acg_ckpt)
    n_chan_use  = args.n_chan_use

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load VAEs ────────────────────────────────────────────────────────────
    print(f"Loading WF VAE from  {wf_ckpt}")
    wf_model = WFConvVAE(n_channels=n_chan_use, n_timepoints=81, latent_dim=LATENT_DIM)
    wf_model.load_state_dict(torch.load(wf_ckpt, map_location=device))
    wf_model.to(device).eval()

    print(f"Loading ACG VAE from {acg_ckpt}")
    acg_model = ACGConvVAE(latent_dim=LATENT_DIM)
    acg_model.load_state_dict(torch.load(acg_ckpt, map_location=device))
    acg_model.to(device).eval()

    # ── Encode all sessions ──────────────────────────────────────────────────
    npz_paths = sorted(dataset_dir.rglob("*_features.npz"))
    if not npz_paths:
        print(f"ERROR: No *_features.npz files found in {dataset_dir}")
        sys.exit(1)
    print(f"\nFound {len(npz_paths)} sessions")

    dfs = []
    for p in npz_paths:
        print(f"  Encoding {p.name} ...", end=" ", flush=True)
        try:
            df = encode_session(p, wf_model, acg_model, n_chan_use, device)
            dfs.append(df)
            print(f"{len(df)} units")
        except Exception as e:
            print(f"SKIP ({e})")

    df_all = pd.concat(dfs, ignore_index=True)
    df_labeled = df_all[df_all["class"] != "unknown"].copy()

    print(f"\nTotal units  : {len(df_all)}")
    print(f"Labeled units: {len(df_labeled)}")
    for cls in CLASSES:
        n = (df_labeled["class"] == cls).sum()
        print(f"  {cls}: {n}")

    # ── Build feature matrix ─────────────────────────────────────────────────
    X = df_labeled[FEATURE_COLS].values.astype(np.float32)
    y = df_labeled["class"].values

    # Replace any NaN with 0
    X = np.nan_to_num(X, nan=0.0)

    # Encode labels to integers (matching notebook approach)
    le = LabelEncoder()
    le.fit(CLASSES)   # fixed order: CF=0, MF=1, PC=2, others=3
    y_enc = le.transform(y)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Class weights for imbalance
    classes_present = np.unique(y)
    cw = compute_class_weight("balanced", classes=classes_present, y=y)
    cw_dict = dict(zip(classes_present, cw))
    sample_weights = np.array([cw_dict[c] for c in y])

    # ── Train XGBoost ────────────────────────────────────────────────────────
    print("\nTraining XGBoost on all labeled data ...")
    clf = xgb.XGBClassifier(
        n_estimators=300,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )
    clf.fit(X_scaled, y_enc, sample_weight=sample_weights)
    print("Done.")

    # Quick sanity check
    y_pred_enc = clf.predict(X_scaled)
    y_pred = le.inverse_transform(y_pred_enc)
    acc = (y_pred == y).mean()
    print(f"Train accuracy (in-sample): {acc:.3f}")

    # ── Save artifacts ───────────────────────────────────────────────────────
    # VAE checkpoints
    shutil.copy(wf_ckpt,  output_dir / "wf_vae.pt")
    shutil.copy(acg_ckpt, output_dir / "acg_vae.pt")
    print(f"Copied VAE checkpoints to {output_dir}")

    # ONNX export — encoder only (returns mu, no sampling needed at inference)
    import torch.nn as nn

    class _MuOnly(nn.Module):
        """Wraps an encoder to return only mu (drop logvar)."""
        def __init__(self, encoder): super().__init__(); self.enc = encoder
        def forward(self, x): mu, _ = self.enc(x); return mu

    wf_onnx  = output_dir / "wf_encoder.onnx"
    acg_onnx = output_dir / "acg_encoder.onnx"

    dummy_wf  = torch.zeros(1, 1, n_chan_use, 81, device=device)
    dummy_acg = torch.zeros(1, 1, 10, 101,       device=device)

    torch.onnx.export(
        _MuOnly(wf_model.encoder).eval(), dummy_wf, str(wf_onnx),
        input_names=["wf"], output_names=["mu"],
        dynamic_axes={"wf": {0: "batch"}, "mu": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported WF encoder  -> {wf_onnx}")

    torch.onnx.export(
        _MuOnly(acg_model.encoder).eval(), dummy_acg, str(acg_onnx),
        input_names=["acg"], output_names=["mu"],
        dynamic_axes={"acg": {0: "batch"}, "mu": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ACG encoder -> {acg_onnx}")

    # XGBoost (native JSON format — version-agnostic)
    clf_path = output_dir / "xgb_clf.json"
    clf.save_model(str(clf_path))
    print(f"Saved XGBoost model -> {clf_path}")

    # StandardScaler
    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler -> {scaler_path}")

    # LabelEncoder
    le_path = output_dir / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"Saved label encoder -> {le_path}")

    # Metadata
    meta = {
        "classes":      CLASSES,
        "feature_cols": FEATURE_COLS,
        "n_chan_use":   n_chan_use,
        "latent_dim":   LATENT_DIM,
        "layer_cats":   LAYER_CATS,
        "n_labeled":    int(len(df_labeled)),
        "class_counts": {c: int((df_labeled["class"] == c).sum()) for c in CLASSES},
    }
    meta_path = output_dir / "clf_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata -> {meta_path}")

    print(f"\nModel folder ready: {output_dir}")
    print("Files:")
    for p in sorted(output_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
