"""
train.py
--------
Training loop for WF and ACG Beta-VAEs.

Typical usage
-------------
See notebooks/train_autoencoders.ipynb for an end-to-end example.

Quick API
---------
    from cerebellum_cell_classifier.autoencoders.models import WFConvVAE, ACGConvVAE
    from cerebellum_cell_classifier.autoencoders.train  import train_vae

    wf_model  = WFConvVAE()
    wf_result = train_vae(wf_model, npz_paths, model_type="wf",
                          save_dir="checkpoints/wf")

    acg_model  = ACGConvVAE()
    acg_result = train_vae(acg_model, npz_paths, model_type="acg",
                           save_dir="checkpoints/acg")
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from .models  import WFConvVAE, ACGConvVAE, elbo_loss
from .datasets import WFDataset, ACGDataset


# ══════════════════════════════════════════════════════════════════════════════
#  Beta schedule
# ══════════════════════════════════════════════════════════════════════════════

def _cosine_beta(epoch: int, beta_max: float, T0: int) -> float:
    """
    CosineAnnealingWarmRestarts beta schedule.
    Ramps from 0 → beta_max over T0 epochs, then restarts.
    """
    cycle = epoch % T0
    return beta_max * 0.5 * (1.0 - math.cos(math.pi * cycle / T0))


# ══════════════════════════════════════════════════════════════════════════════
#  Main training function
# ══════════════════════════════════════════════════════════════════════════════

def train_vae(
    model:             "WFConvVAE | ACGConvVAE",
    npz_paths:         list[str | Path],
    *,
    model_type:        str   = "wf",
    val_fraction:      float = 0.1,
    n_epochs:          int   = 150,
    batch_size:        int   = 64,
    lr:                float = 1e-3,
    beta_max:          float = 5.0,
    beta_anneal:       bool  = False,
    T0:                int   = 20,
    short_lag_weight:  float = 1.5,
    n_short:           int   = 20,
    amplitude_weight:  float = 0.0,
    min_spikes:        int   = 0,
    n_chan_use:        "int | None" = None,
    patience:          "int | None" = None,
    save_dir:          "str | Path | None" = None,
    device:            "str | None" = None,
    verbose:           bool  = True,
) -> dict:
    """
    Train a WF or ACG Beta-VAE on features loaded from multiple NPZ files.

    Parameters
    ----------
    model         : WFConvVAE or ACGConvVAE instance
    npz_paths     : list of .npz paths produced by run_extraction.py
    model_type    : ``"wf"`` or ``"acg"`` — selects Dataset and loss settings
    val_fraction  : fraction of data held out for validation (default 0.1)
    n_epochs      : maximum number of training epochs
                    (Cell 2025 defaults: 150 for WF, 60 for ACG)
    batch_size    : mini-batch size  (64 for WF, 32 for ACG)
    lr            : Adam learning rate  (1e-3 for WF, 5e-4 for ACG)
    beta_max      : maximum / constant KL weight (default 5)
    beta_anneal   : if True use cosine-annealing beta schedule (ACG);
                    if False use constant beta_max from epoch 0 (WF)
    T0            : restart period for cosine beta schedule (default 20)
    short_lag_weight : extra MSE weight on the first ``n_short`` lag bins
                       of the ACG reconstruction (default 1.5; ignored for WF)
    n_short       : number of short-lag bins to up-weight (default 20)
    amplitude_weight : WF only — per-sample MSE weight = 1 + amplitude_weight*|target|,
                       so spike-peak samples dominate the loss.  0 = flat MSE (default).
                       Typical value: 3.0–5.0.
    min_spikes    : skip units whose n_spikes_wf is below this value
    n_chan_use    : keep only the first N channels of the waveform (WF only).
                   None = use all channels.  e.g. 4 → keep 4 main channels
    patience      : early-stopping patience in epochs — stop if val loss does
                    not improve for this many consecutive epochs, then restore
                    the best weights.  None = disabled (default)
    save_dir      : directory to save best + final checkpoints; None = no save
    device        : ``'cuda'``, ``'cpu'``, or None (auto-detect)
    verbose       : print loss every 10 epochs

    Returns
    -------
    dict
        train_losses  : list of per-epoch training loss
        val_losses    : list of per-epoch validation loss
        best_epoch    : epoch with lowest validation loss
        best_val_loss : best validation loss value
    """
    # ── Device ───────────────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model = model.to(dev)
    if verbose:
        print(f"Training on: {dev}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if model_type == "wf":
        dataset    = WFDataset(npz_paths, min_spikes=min_spikes, n_chan_use=n_chan_use)
        w_short    = 1.0    # no lag-weighting for waveforms
    elif model_type == "acg":
        dataset    = ACGDataset(npz_paths, min_spikes=min_spikes)
        w_short    = short_lag_weight
    else:
        raise ValueError(f"model_type must be 'wf' or 'acg', got '{model_type}'")

    n_val   = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    if verbose:
        print(f"Train: {n_train} units  |  Val: {n_val} units")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=0, pin_memory=(dev.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=0, pin_memory=(dev.type == "cuda"),
    )

    # ── Optimizer + LR scheduler ─────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # ReduceLROnPlateau: halves LR when val loss hasn't improved for 10 epochs.
    # No hard restarts → smooth val curve → reliable best-epoch selection.
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val_loss  = float("inf")
    best_epoch     = 0
    best_state     = None   # in-memory copy of best weights for early stopping
    no_improve_cnt = 0      # epochs without improvement

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        beta = _cosine_beta(epoch - 1, beta_max, T0) if beta_anneal else beta_max

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        t_total = 0.0; r_total = 0.0; k_total = 0.0
        for batch in train_loader:
            batch = batch.to(dev)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, r, k = elbo_loss(
                recon, batch, mu, logvar,
                beta=beta, short_lag_weight=w_short, n_short=n_short,
                amplitude_weight=amplitude_weight,
            )
            loss.backward()
            optimizer.step()
            t_total += loss.item(); r_total += r; k_total += k

        train_loss = t_total / len(train_loader)
        train_recon = r_total / len(train_loader)
        train_kl    = k_total / len(train_loader)
        train_losses.append(train_loss)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        v_total = 0.0; vr_total = 0.0; vk_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(dev)
                recon, mu, logvar = model(batch)
                loss, r, k = elbo_loss(
                    recon, batch, mu, logvar,
                    beta=beta_max,            # fixed beta for val
                    short_lag_weight=w_short,
                    n_short=n_short,
                    amplitude_weight=amplitude_weight,
                )
                v_total += loss.item(); vr_total += r; vk_total += k

        val_loss  = v_total  / len(val_loader)
        val_recon = vr_total / len(val_loader)
        val_kl    = vk_total / len(val_loader)
        val_losses.append(val_loss)
        lr_sched.step(val_loss)   # reduce LR when val loss stalls

        # Save best checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss  = val_loss
            best_epoch     = epoch
            no_improve_cnt = 0
            # keep an in-memory copy so we can restore even without save_dir
            import copy
            best_state = copy.deepcopy(model.state_dict())
            if save_dir is not None:
                torch.save(best_state, save_dir / f"{model_type}_vae_best.pt")
        else:
            no_improve_cnt += 1

        if verbose and (epoch % 10 == 0 or epoch == 1 or is_best):
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{n_epochs}  beta={beta:.3f}  lr={cur_lr:.2e}  "
                  f"train={train_loss:.4f} (recon={train_recon:.4f} kl={train_kl:.4f})  "
                  f"val={val_loss:.4f} (recon={val_recon:.4f} kl={val_kl:.4f})"
                  + ("  *best*" if is_best else ""))

        # Early stopping
        if patience is not None and no_improve_cnt >= patience:
            if verbose:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
            break

    # Restore best weights so the returned model is always the best one
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save final model (= best weights after early stopping or full run)
    if save_dir is not None:
        torch.save(model.state_dict(),
                   save_dir / f"{model_type}_vae_final.pt")
        if verbose:
            print(f"\n  Best epoch: {best_epoch}  "
                  f"(val loss {best_val_loss:.4f})")
            print(f"  Checkpoints saved to: {save_dir}")

    return {
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
    }
