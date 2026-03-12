"""
plots_panel.py
--------------
Three-panel viewer:
  Top left  : 8-channel waveform
  Top right : 1-D ACG bar chart
  Bottom    : 3-D ACG heatmap (log-lag x FR-quantile bin)

Channel order: ch 0 = main channel (always), ch 1-7 sorted by
physical distance from main (closest first).  NOT sorted by amplitude.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QSplitter, QVBoxLayout
from PyQt5.QtCore import Qt

BG  = "#16213e"
FG  = "#e0e0e0"

WF_MAIN       = "#4fc3f7"   # main channel
WF_OTHER      = "#4a90d9"   # neighbour channels
WF_FILL_ALPHA = 40          # 0-255
ACG_COLOR     = "#2196F3"
PEAK_SAMPLE   = 40          # centre of the 81-sample window


def _styled_plot(gw, title=""):
    p = gw.addPlot(title=title)
    p.getViewBox().setBackgroundColor(BG)
    for ax in ("bottom", "left"):
        p.getAxis(ax).setPen(FG)
        p.getAxis(ax).setTextPen(FG)
    if title:
        p.setTitle(title, color=FG, size="9pt")
    return p


def _jet_lut():
    """Build a jet-like lookup table using pyqtgraph's colormap."""
    try:
        # getLookupTable(start, stop, nPts) -- must use keyword or positional args
        return pg.colormap.get("jet", source="matplotlib").getLookupTable(0.0, 1.0, 256)
    except Exception:
        try:
            return pg.colormap.get("hot", source="matplotlib").getLookupTable(0.0, 1.0, 256)
        except Exception:
            return None


class PlotsPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wf_scale    = 1.0
        self._wf_norm     = True    # normalised by default
        self._wf_half_win = 40      # samples either side of peak (40 = full window)
        self._acg_xlim    = 20.0
        self._acg3d_clim  = None
        self._acg3d_log   = True    # log X axis (True) or linear (False)
        self._acg3d_xlim  = 200.0   # max lag displayed on X axis (ms)

        # Persistent pyqtgraph items (re-used across unit changes)
        self._wf_lines: list[pg.PlotCurveItem] = []
        self._wf_fills: list[pg.PlotCurveItem] = []
        self._wf_n_ch  = -1
        self._acg_bars = None

        self._build_ui()

    # ── Layout ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Top row: WF | 1D ACG
        top = QSplitter(Qt.Horizontal)

        wf_gw = pg.GraphicsLayoutWidget()
        wf_gw.setBackground(BG)
        self._wf_plot = _styled_plot(wf_gw, "Waveform")
        self._wf_plot.hideAxis("left")
        self._wf_plot.setLabel("bottom", "Sample", color=FG)
        top.addWidget(wf_gw)

        acg_gw = pg.GraphicsLayoutWidget()
        acg_gw.setBackground(BG)
        self._acg_plot = _styled_plot(acg_gw, "1D ACG")
        self._acg_plot.setLabel("bottom", "Lag (ms)", color=FG)
        self._acg_plot.setLabel("left", "Rate (Hz)", color=FG)
        top.addWidget(acg_gw)

        top.setStretchFactor(0, 2)
        top.setStretchFactor(1, 3)

        # Bottom: 3D ACG heatmap
        acg3d_gw = pg.GraphicsLayoutWidget()
        acg3d_gw.setBackground(BG)
        self._acg3d_plot = _styled_plot(acg3d_gw, "3D ACG (log-lag x FR-quantile)")
        self._acg3d_plot.setLabel("bottom", "Lag (ms, log)", color=FG)
        self._acg3d_plot.setLabel("left", "FR quantile bin", color=FG)

        self._acg3d_img = pg.ImageItem()
        lut = _jet_lut()
        if lut is not None:
            self._acg3d_img.setLookupTable(lut)
        self._acg3d_plot.addItem(self._acg3d_img)

        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(top)
        vsplit.addWidget(acg3d_gw)
        vsplit.setStretchFactor(0, 2)
        vsplit.setStretchFactor(1, 1)

        outer.addWidget(vsplit)

    # ── Public update ───────────────────────────────────────────────────────────
    def update_unit(self, i: int, data):
        mean, std, depth = data.get_wf(i)
        acg, t_ms        = data.get_acg_1d(i)
        acg3d, t_log     = data.get_acg_3d(i)
        uid   = int(data.unit_ids[i])
        label = data.get_label(i)

        norm_str = "norm" if self._wf_norm else "raw"
        self._wf_plot.setTitle(
            f"Waveform  unit {uid}  [{label}]  ({norm_str})", color=FG, size="9pt")
        self._acg_plot.setTitle(f"1D ACG  unit {uid}", color=FG, size="9pt")
        mode = "log" if self._acg3d_log else "linear"
        self._acg3d_plot.setTitle(f"3D ACG  unit {uid}  [{mode}]", color=FG, size="9pt")

        self._draw_wf(mean, std)
        self._draw_acg_1d(acg, t_ms)
        self._draw_acg_3d(acg3d, t_log)

    # ── Waveform ────────────────────────────────────────────────────────────────
    def _draw_wf(self, mean: np.ndarray, std: np.ndarray):
        """
        mean, std : (n_ch, n_samples) float32
        ch 0 = main channel; ch 1-7 sorted by distance (not amplitude).
        Window: PEAK_SAMPLE ± _wf_half_win samples.
        """
        n_ch, n_t = mean.shape

        # Crop to requested window around peak
        hw   = self._wf_half_win
        s0   = max(0, PEAK_SAMPLE - hw)
        s1   = min(n_t, PEAK_SAMPLE + hw + 1)
        mean = mean[:, s0:s1]
        std  = std[:, s0:s1]
        n_t  = mean.shape[1]
        t    = np.arange(s0, s0 + n_t, dtype=np.float64)

        # Normalise (or not)
        if self._wf_norm:
            # Divide ALL channels by the peak-to-trough of the MAIN channel (ch 0)
            amp = float(mean[0].max() - mean[0].min())
            if amp == 0:
                amp = float(np.abs(mean).max()) or 1.0
            mn = mean / amp
            sd = std  / amp
        else:
            mn = mean.astype(np.float64)
            sd = std.astype(np.float64)

        scale = self._wf_scale
        # spacing is fixed relative to data amplitude so scale only changes
        # waveform height, not the inter-channel separation
        if self._wf_norm:
            spacing = 1.5   # normalized amplitude ≈ 1.0 for main channel
        else:
            base_amp = float(mn[0].max() - mn[0].min())
            if base_amp == 0:
                base_amp = float(np.abs(mn).max()) or 1.0
            spacing = 1.5 * base_amp

        # Re-allocate if channel count changed
        if n_ch != self._wf_n_ch:
            self._wf_plot.clear()
            self._wf_lines = []
            self._wf_fills = []
            fill_c = pg.mkColor(WF_OTHER)
            fill_c.setAlpha(WF_FILL_ALPHA)
            for ch in range(n_ch):
                color = WF_MAIN if ch == 0 else WF_OTHER
                width = 2.0    if ch == 0 else 1.2
                fill = pg.PlotCurveItem(
                    pen=pg.mkPen(None), brush=pg.mkBrush(fill_c), antialias=True)
                self._wf_plot.addItem(fill)
                self._wf_fills.append(fill)
                line = pg.PlotCurveItem(
                    pen=pg.mkPen(color=color, width=width), antialias=True)
                self._wf_plot.addItem(line)
                self._wf_lines.append(line)
            self._wf_n_ch = n_ch

        for ch in range(n_ch):
            y_off  = ch * spacing
            m      = mn[ch] * scale
            s      = sd[ch] * scale
            tf     = np.concatenate([t, t[::-1]])
            yf     = np.concatenate([m + s + y_off, (m - s + y_off)[::-1]])
            self._wf_fills[ch].setData(tf, yf)
            self._wf_lines[ch].setData(t, m + y_off)

        self._wf_plot.setXRange(t[0], t[-1], padding=0.02)
        self._wf_plot.setYRange(-spacing, (n_ch - 1 + 1.5) * spacing, padding=0.02)

    # ── 1D ACG ──────────────────────────────────────────────────────────────────
    def _draw_acg_1d(self, acg: np.ndarray, t_ms: np.ndarray):
        xlim  = self._acg_xlim
        mask  = np.abs(t_ms) <= xlim
        t_z   = t_ms[mask]
        a_z   = acg[mask]
        bin_w = float(t_ms[1] - t_ms[0])

        if self._acg_bars is None:
            self._acg_bars = pg.BarGraphItem(
                x=t_z, height=a_z, width=bin_w,
                brush=pg.mkBrush(ACG_COLOR), pen=pg.mkPen(None),
            )
            self._acg_plot.addItem(self._acg_bars)
        else:
            self._acg_bars.setOpts(x=t_z, height=a_z, width=bin_w)

        self._acg_plot.setXRange(-xlim - 0.5, xlim + 0.5, padding=0)
        self._acg_plot.setYRange(0, float(a_z.max()) * 1.12 + 0.5, padding=0)

    # ── 3D ACG ──────────────────────────────────────────────────────────────────
    def _draw_acg_3d(self, acg3d: np.ndarray, t_log: np.ndarray):
        """
        acg3d : (201, 10)  axis 0 = log-lag bins, axis 1 = FR-quantile bins
        t_log : (201,)     lag axis in ms (log-spaced, symmetric around 0)

        Log mode  : bins keep equal screen width; custom ticks label the real
                    ms values so the axis reads as a log scale.
        Linear mode: data is resampled onto a uniform ms grid; axis reads ms.
        """
        n_lag, n_fr = acg3d.shape
        xlim = self._acg3d_xlim

        # vmax from full data so colour scale is stable when panning/zooming
        vmax = self._acg3d_clim
        if vmax is None:
            vmax = float(np.percentile(acg3d, 99))
        vmax = max(vmax, 1e-6)

        if self._acg3d_log:
            # ── Log mode ────────────────────────────────────────────────────
            # Crop to ±xlim, keeping the original log-spaced bin structure
            mask   = np.abs(t_log) <= xlim
            if not np.any(mask):
                return
            t_show   = t_log[mask]
            img_show = acg3d[mask].astype(np.float32)
            n_pix    = img_show.shape[0]
            t_min    = float(t_show[0])
            t_max    = float(t_show[-1])
            width    = t_max - t_min

            self._acg3d_img.setImage(img_show, autoLevels=False, levels=(0, vmax))
            self._acg3d_img.setRect(pg.QtCore.QRectF(t_min, 0, width, float(n_fr)))
            self._acg3d_plot.setXRange(t_min, t_max, padding=0.02)
            self._acg3d_plot.setLabel("bottom", "Lag (ms, log)", color=FG)
            # Relabel axis ticks at round ms values
            self._set_log_x_ticks(t_show, t_min, width)

        else:
            # ── Linear mode ─────────────────────────────────────────────────
            # Resample log-binned data onto a uniform ms grid
            t_lin   = np.linspace(-xlim, xlim, 201)
            idx     = np.searchsorted(t_log, t_lin).clip(0, n_lag - 1)
            img_lin = acg3d[idx].astype(np.float32)

            self._acg3d_img.setImage(img_lin, autoLevels=False, levels=(0, vmax))
            self._acg3d_img.setRect(
                pg.QtCore.QRectF(-xlim, 0, 2.0 * xlim, float(n_fr)))
            self._acg3d_plot.setXRange(-xlim, xlim, padding=0.02)
            self._acg3d_plot.setLabel("bottom", "Lag (ms)", color=FG)
            # Restore auto ticks
            self._acg3d_plot.getAxis("bottom").setTicks(None)

        self._acg3d_plot.setYRange(0, n_fr, padding=0.02)

    def _set_log_x_ticks(self, t_show: np.ndarray, t_min: float, width: float):
        """Place x-axis ticks at round ms values for log-scale display.

        Each log bin occupies equal screen width, so the scene coordinate of
        bin i is:  t_min + (i + 0.5) * width / n_bins
        We pick bins closest to round ms values and label them accordingly.
        """
        n = len(t_show)
        if n < 2 or width == 0:
            return

        round_ms = [0.8, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
        candidates = [0.0] + [-ms for ms in round_ms] + list(round_ms)

        ticks = []
        seen = set()
        for ms in candidates:
            if not (t_show[0] <= ms <= t_show[-1]):
                continue
            idx = int(np.argmin(np.abs(t_show - ms)))
            scene_x = t_min + (idx + 0.5) * width / n
            key = round(scene_x, 1)
            if key in seen:
                continue
            seen.add(key)
            if ms == 0:
                label = "0"
            elif ms > 0:
                label = f"{int(ms)}" if ms >= 1 else f"{ms:.1f}"
            else:
                label = f"{int(ms)}" if ms <= -1 else f"{ms:.1f}"
            ticks.append((scene_x, label))

        ticks.sort()
        self._acg3d_plot.getAxis("bottom").setTicks([ticks, []])

    # ── External setters (called by controls) ────────────────────────────────────
    def set_wf_scale(self, v: float):
        self._wf_scale = max(0.05, v)

    def set_wf_norm(self, norm: bool):
        self._wf_norm = norm
        # Force re-allocation of plot items (y-axis unit changes)
        self._wf_n_ch = -1

    def set_wf_half_win(self, hw: int):
        self._wf_half_win = max(1, hw)

    def set_acg_xlim(self, v: float):
        self._acg_xlim = max(1.0, v)

    def set_acg3d_clim(self, v):
        self._acg3d_clim = v

    def set_acg3d_log(self, log: bool):
        self._acg3d_log = log

    def set_acg3d_xlim(self, xlim: float):
        self._acg3d_xlim = max(1.0, xlim)
