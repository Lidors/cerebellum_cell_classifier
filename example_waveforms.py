"""
example_waveforms.py
--------------------
Minimal example: extract mean waveforms for a set of manually labelled units
from a Neuropixels session.

Usage:
    python example_waveforms.py --session /path/to/session --n_ch_total 385
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from cerebellum_cell_classifier.io.kilosort import load_kilosort
from cerebellum_cell_classifier.features.waveform import (
    build_waveform_features,
    normalize_waveforms,
)


def main(session_path: str, n_channels_total: int = 385) -> None:

    # ── 1. Load Kilosort output ─────────────────────────────────────────────
    print("Loading Kilosort data ...")
    ks = load_kilosort(session_path)
    print(f"  {len(ks.good_units)} good units found")

    # ── 2. Define units to process + optional expert labels ─────────────────
    #   Replace this with your own unit list / label dict.
    unit_ids = ks.good_units          # process all good units

    # Example expert labels  {cluster_id: cell_type_string}
    # Supported strings: 'PC', 'CF', 'MLI', 'GC', 'UBC', 'MF', 'unknown'
    cell_type_labels = {}             # fill in your annotations here

    # ── 3. Extract mean waveforms ────────────────────────────────────────────
    result = build_waveform_features(
        session_path=session_path,
        ks_data=ks,
        unit_ids=unit_ids,
        cell_type_labels=cell_type_labels,
        n_channels_total=n_channels_total,
        n_channels_extract=8,
        n_samples=81,
        peak_sample=40,     # 0-based centre of 81-sample window
        max_spikes=3_000,
        do_realign=True,
    )

    print(f"\nExtracted waveforms for {len(result['unit_ids'])} units")
    print(f"  mean_waveforms shape : {result['mean_waveforms'].shape}")
    print(f"  std_waveforms  shape : {result['std_waveforms'].shape}")

    # ── 4. Normalise (amplitude = 1, negative primary deflection) ───────────
    norm_wfs = normalize_waveforms(result["mean_waveforms"])

    # ── 5. Quick sanity plot: first 9 units ─────────────────────────────────
    n_plot = min(9, len(result["unit_ids"]))
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    t_ms = (np.arange(81) - 40) / 30.0   # time axis in ms  (30 kHz)

    for ax, i in zip(axes.flat, range(n_plot)):
        uid   = result["unit_ids"][i]
        label = result["cell_type_labels"][i]
        wf    = norm_wfs[i]           # (8 channels, 81 samples)
        # Offset channels vertically by probe position (main ch at top)
        for ch in range(8):
            ax.plot(t_ms, wf[ch] + ch * 1.5, lw=0.8, color="steelblue")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"unit {uid}  [{label}]", fontsize=8)
        ax.set_xlabel("time (ms)", fontsize=7)
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("waveforms_preview.png", dpi=150)
    print("\nSaved waveforms_preview.png")
    plt.show()

    # ── 6. Save results as .npz ──────────────────────────────────────────────
    out_path = "waveform_features.npz"
    np.savez(
        out_path,
        unit_ids=result["unit_ids"],
        cell_type_labels=result["cell_type_labels"],
        mean_waveforms=result["mean_waveforms"],
        std_waveforms=result["std_waveforms"],
        main_channels=result["main_channels"],
        used_channels=result["used_channels"],
        norm_waveforms=norm_wfs,
    )
    print(f"Saved features to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="Path to session directory")
    parser.add_argument("--n_ch_total", type=int, default=385,
                        help="Total channels in binary file (default: 385)")
    args = parser.parse_args()
    main(args.session, args.n_ch_total)
