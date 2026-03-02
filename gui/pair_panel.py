"""
pair_panel.py
-------------
Pair explorer widget for inspecting CCG-detected pairs.

Layout
------
  Left   : sortable pair table (detected pairs first, then undetected)
  Centre : 5 plots — WF_A | CCG | WF_B (top)  ACG_A | ACG_B (bottom)
  Right  : pair info + label dropdowns + Accept / Reject buttons

Keyboard shortcuts (active when pair view is focused)
-----------------------------------------------------
  Left / Right  — previous / next pair
  A             — accept CCG labels (copy to manual labels)
  R             — reject pair (set both manual labels to 'unknown')
  N / P         — next / previous pair for the *same unit*
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QAbstractItemView,
    QHeaderView, QPushButton, QLabel, QComboBox, QGroupBox,
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor, QFont

BG = "#16213e"
FG = "#e0e0e0"

_BTN = (
    "QPushButton { background: #1a1a3e; color: #ccccee; "
    "border: 1px solid #3a3a6a; border-radius: 3px; padding: 4px 12px; font-size: 9pt; }"
    "QPushButton:hover { background: #2a2a6a; }"
)
_BTN_ACCEPT = (
    "QPushButton { background: #1b5e20; color: #c8e6c9; "
    "border: 1px solid #2e7d32; border-radius: 3px; padding: 4px 12px; font-size: 9pt; }"
    "QPushButton:hover { background: #2e7d32; }"
)
_BTN_REJECT = (
    "QPushButton { background: #b71c1c; color: #ffcdd2; "
    "border: 1px solid #c62828; border-radius: 3px; padding: 4px 12px; font-size: 9pt; }"
    "QPushButton:hover { background: #c62828; }"
)
_GRP = (
    "QGroupBox { color: #8888cc; border: 1px solid #2a2a5a; "
    "border-radius: 4px; margin-top: 8px; padding: 4px; font-size: 8pt; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
)

VALID_LABELS = ["PC", "CF", "MLI", "GC", "UBC", "MF", "GoC", "unknown"]

_TYPE_COLORS = {
    "pc_cf": "#4fc3f7",
    "mli":   "#E91E63",
    "none":  "#555555",
}

WF_MAIN   = "#4fc3f7"
WF_OTHER  = "#4a90d9"
ACG_COLOR = "#2196F3"
CCG_COLOR = "#66bb6a"
PEAK_SAMPLE = 40


def _styled_plot(gw, title=""):
    p = gw.addPlot(title=title)
    p.getViewBox().setBackgroundColor(BG)
    for ax in ("bottom", "left"):
        p.getAxis(ax).setPen(FG)
        p.getAxis(ax).setTextPen(FG)
    if title:
        p.setTitle(title, color=FG, size="9pt")
    return p


class _NumItem(QTableWidgetItem):
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return self.text() < other.text()


class PairPanel(QWidget):
    """Full pair explorer widget (table + plots + controls)."""

    # Emitted when user wants to go back to the unit view for a specific unit
    go_to_unit = pyqtSignal(int)        # unit array index
    labels_changed = pyqtSignal()       # after accept / reject

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._pair_order: np.ndarray = np.array([], dtype=np.int64)
        self._cur_pair_pos = -1   # position in _pair_order
        self._unit_pair_idxs: np.ndarray | None = None  # for N/P shortcuts
        self._unit_pair_pos = -1
        self._wf_scale = 1.0   # Ctrl+Up/Down to zoom WF amplitude
        self._flip_ccg = False
        self._build_ui()

    # ══════════════════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # ── Left: pair table ─────────────────────────────────────────────
        table_w = QWidget()
        tlay = QVBoxLayout(table_w)
        tlay.setContentsMargins(4, 4, 4, 4)
        tlay.setSpacing(4)

        self._count_label = QLabel("0 pairs")
        self._count_label.setStyleSheet("color: #8888aa; font-size: 8pt;")
        tlay.addWidget(self._count_label)

        # Filter buttons
        frow = QHBoxLayout()
        for lbl in ("All", "Detected", "PC/CF", "MLI"):
            btn = QPushButton(lbl)
            btn.setFixedHeight(22)
            btn.setStyleSheet(
                "QPushButton { background:#1a1a3e; color:#aaa; border:1px solid #333; "
                "border-radius:2px; padding:0 6px; font-size:8pt; }"
                "QPushButton:hover { background:#2a2a6a; color:#eee; }"
            )
            btn.clicked.connect(lambda _, t=lbl: self._filter_pairs(t))
            frow.addWidget(btn)
        tlay.addLayout(frow)

        cols = ["#", "Unit A", "Unit B", "Type", "Score", "Dist"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setDefaultSectionSize(20)
        self._table.setFont(QFont("", 8))
        self._table.itemSelectionChanged.connect(self._on_table_select)
        tlay.addWidget(self._table)

        splitter.addWidget(table_w)

        # ── Centre: plots ────────────────────────────────────────────────
        plots_w = QWidget()
        play = QVBoxLayout(plots_w)
        play.setContentsMargins(0, 0, 0, 0)

        # Top row: WF_A | CCG | WF_B
        top_split = QSplitter(Qt.Horizontal)

        gw_wfA = pg.GraphicsLayoutWidget(); gw_wfA.setBackground(BG)
        self._wfA_plot = _styled_plot(gw_wfA, "WF Unit A")
        self._wfA_plot.hideAxis("left")
        top_split.addWidget(gw_wfA)

        gw_ccg = pg.GraphicsLayoutWidget(); gw_ccg.setBackground(BG)
        self._ccg_plot = _styled_plot(gw_ccg, "CCG")
        self._ccg_plot.setLabel("bottom", "Lag (ms)", color=FG)
        self._ccg_plot.setLabel("left", "Counts", color=FG)
        top_split.addWidget(gw_ccg)

        gw_wfB = pg.GraphicsLayoutWidget(); gw_wfB.setBackground(BG)
        self._wfB_plot = _styled_plot(gw_wfB, "WF Unit B")
        self._wfB_plot.hideAxis("left")
        top_split.addWidget(gw_wfB)

        top_split.setStretchFactor(0, 1)
        top_split.setStretchFactor(1, 2)
        top_split.setStretchFactor(2, 1)

        # Bottom row: ACG_A | ACG_B
        bot_split = QSplitter(Qt.Horizontal)

        gw_acgA = pg.GraphicsLayoutWidget(); gw_acgA.setBackground(BG)
        self._acgA_plot = _styled_plot(gw_acgA, "ACG Unit A")
        self._acgA_plot.setLabel("bottom", "Lag (ms)", color=FG)
        self._acgA_plot.setLabel("left", "Rate (Hz)", color=FG)
        bot_split.addWidget(gw_acgA)

        gw_acgB = pg.GraphicsLayoutWidget(); gw_acgB.setBackground(BG)
        self._acgB_plot = _styled_plot(gw_acgB, "ACG Unit B")
        self._acgB_plot.setLabel("bottom", "Lag (ms)", color=FG)
        self._acgB_plot.setLabel("left", "Rate (Hz)", color=FG)
        bot_split.addWidget(gw_acgB)

        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(top_split)
        vsplit.addWidget(bot_split)
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 2)
        play.addWidget(vsplit)

        splitter.addWidget(plots_w)

        # ── Right: controls ──────────────────────────────────────────────
        ctrl_w = QWidget()
        ctrl_w.setMinimumWidth(170)
        ctrl_w.setMaximumWidth(220)
        clay = QVBoxLayout(ctrl_w)
        clay.setContentsMargins(6, 6, 6, 6)
        clay.setSpacing(6)
        clay.setAlignment(Qt.AlignTop)

        # Navigation
        nav_grp = QGroupBox("Navigation")
        nav_grp.setStyleSheet(_GRP)
        nav_lay = QHBoxLayout(nav_grp)
        self._prev_btn = QPushButton("< Prev")
        self._next_btn = QPushButton("Next >")
        self._prev_btn.setStyleSheet(_BTN)
        self._next_btn.setStyleSheet(_BTN)
        self._prev_btn.clicked.connect(self._go_prev)
        self._next_btn.clicked.connect(self._go_next)
        nav_lay.addWidget(self._prev_btn)
        nav_lay.addWidget(self._next_btn)
        clay.addWidget(nav_grp)

        # Pair info
        self._pair_info = QLabel("No pair selected")
        self._pair_info.setWordWrap(True)
        self._pair_info.setStyleSheet("color: #aaaacc; font-size: 9pt;")
        clay.addWidget(self._pair_info)

        # Label editors
        lbl_grp = QGroupBox("Labels")
        lbl_grp.setStyleSheet(_GRP)
        llay = QVBoxLayout(lbl_grp)
        llay.setSpacing(4)

        llay.addWidget(QLabel("Unit A:"))
        self._lbl_a = QComboBox()
        self._lbl_a.addItems(VALID_LABELS)
        llay.addWidget(self._lbl_a)

        llay.addWidget(QLabel("Unit B:"))
        self._lbl_b = QComboBox()
        self._lbl_b.addItems(VALID_LABELS)
        llay.addWidget(self._lbl_b)

        clay.addWidget(lbl_grp)

        # Accept / Reject
        self._accept_btn = QPushButton("Accept (A)")
        self._accept_btn.setStyleSheet(_BTN_ACCEPT)
        self._accept_btn.setToolTip("Copy CCG-suggested labels into manual labels")
        self._accept_btn.clicked.connect(self._accept_pair)
        clay.addWidget(self._accept_btn)

        self._reject_btn = QPushButton("Reject (R)")
        self._reject_btn.setStyleSheet(_BTN_REJECT)
        self._reject_btn.setToolTip("Set both manual labels to 'unknown'")
        self._reject_btn.clicked.connect(self._reject_pair)
        clay.addWidget(self._reject_btn)

        # Go to unit buttons
        clay.addSpacing(12)
        self._goto_a_btn = QPushButton("Unit A in Units tab")
        self._goto_a_btn.setStyleSheet(_BTN)
        self._goto_a_btn.clicked.connect(self._goto_unit_a)
        clay.addWidget(self._goto_a_btn)

        self._goto_b_btn = QPushButton("Unit B in Units tab")
        self._goto_b_btn.setStyleSheet(_BTN)
        self._goto_b_btn.clicked.connect(self._goto_unit_b)
        clay.addWidget(self._goto_b_btn)

        clay.addStretch()

        # Keyboard hint
        hint = QLabel(
            "<b>Keys:</b> Left/Right = prev/next pair<br>"
            "A = accept, R = reject<br>"
            "N/P = next/prev pair for same unit<br>"
            "Ctrl+Up/Down = zoom WF"
        )
        hint.setStyleSheet("color: #6666aa; font-size: 7pt;")
        hint.setWordWrap(True)
        clay.addWidget(hint)

        splitter.addWidget(ctrl_w)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([240, 800, 190])

        # Persistent bar items
        self._ccg_bars = None
        self._acgA_bars = None
        self._acgB_bars = None

    # ══════════════════════════════════════════════════════════════════════════
    #  Data binding
    # ══════════════════════════════════════════════════════════════════════════
    def set_data(self, data):
        """Bind a SessionData instance and populate the table."""
        self._data = data
        self._populate_table()

    def _populate_table(self, type_filter: str = "All"):
        if self._data is None or not self._data.has_ccg:
            self._pair_order = np.array([], dtype=np.int64)
            self._table.setRowCount(0)
            self._count_label.setText("No CCG data")
            return

        all_idxs = self._data.get_all_pair_indices_sorted()

        if type_filter == "Detected":
            all_idxs = all_idxs[self._data.ccg_pair_types[all_idxs] != "none"]
        elif type_filter == "PC/CF":
            all_idxs = all_idxs[self._data.ccg_pair_types[all_idxs] == "pc_cf"]
        elif type_filter == "MLI":
            all_idxs = all_idxs[self._data.ccg_pair_types[all_idxs] == "mli"]

        self._pair_order = all_idxs
        d = self._data

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(all_idxs))

        for row, pidx in enumerate(all_idxs):
            uid_a, uid_b = d.get_pair_uids(pidx)
            ptype  = str(d.ccg_pair_types[pidx])
            score  = float(d.ccg_pair_scores[pidx])
            dist   = float(d.ccg_pair_dists[pidx])

            cells = [
                _NumItem(str(row)),
                _NumItem(str(uid_a)),
                _NumItem(str(uid_b)),
                QTableWidgetItem(ptype),
                _NumItem(f"{score:.2f}"),
                _NumItem(f"{dist:.0f}"),
            ]
            for col, item in enumerate(cells):
                item.setTextAlignment(Qt.AlignCenter)
                item.setData(Qt.UserRole, int(pidx))
                if col == 3:
                    color = _TYPE_COLORS.get(ptype, "#888")
                    item.setForeground(QColor(color))
                self._table.setItem(row, col, item)

        self._table.setSortingEnabled(True)
        n_det = int((d.ccg_pair_types[all_idxs] != "none").sum()) if len(all_idxs) else 0
        self._count_label.setText(
            f"{len(all_idxs)} pairs ({n_det} detected) — filter: {type_filter}")

    def _filter_pairs(self, type_filter: str):
        self._populate_table(type_filter)

    # ══════════════════════════════════════════════════════════════════════════
    #  Pair loading
    # ══════════════════════════════════════════════════════════════════════════
    def load_pair(self, pair_idx: int):
        """Load and display a specific pair by its pair index."""
        if self._data is None or pair_idx < 0 or pair_idx >= self._data.n_pairs:
            return

        d = self._data
        uid_a, uid_b = d.get_pair_uids(pair_idx)
        idx_a = d.uid_to_idx(uid_a)
        idx_b = d.uid_to_idx(uid_b)
        if idx_a < 0 or idx_b < 0:
            return

        ptype = str(d.ccg_pair_types[pair_idx])
        score = float(d.ccg_pair_scores[pair_idx])
        dist  = float(d.ccg_pair_dists[pair_idx])

        # ── Orientation: key unit (PC / inhibited target) always on RIGHT ──
        ccg_lbl_a0 = d.get_ccg_label(idx_a)
        ccg_lbl_b0 = d.get_ccg_label(idx_b)
        flip_ccg = (
            (ptype == "pc_cf" and ccg_lbl_a0 == "PC") or
            (ptype == "mli"   and ccg_lbl_a0 != "MLI" and ccg_lbl_b0 == "MLI")
        )
        if flip_ccg:
            uid_a, uid_b = uid_b, uid_a
            idx_a, idx_b = idx_b, idx_a
        self._flip_ccg = flip_ccg
        # ───────────────────────────────────────────────────────────────────

        # Update pair info
        lbl_a = d.get_label(idx_a)
        lbl_b = d.get_label(idx_b)
        ccg_a = d.get_ccg_label(idx_a)
        ccg_b = d.get_ccg_label(idx_b)
        fr_a  = d.get_mean_fr(idx_a)
        fr_b  = d.get_mean_fr(idx_b)

        self._pair_info.setText(
            f"<b>Pair: {uid_a} \u2194 {uid_b}</b><br>"
            f"Type: <b>{ptype}</b>  Score: {score:.2f}<br>"
            f"Dist: {dist:.0f} \u00b5m<br><br>"
            f"Unit A ({uid_a}): {lbl_a}  FR={fr_a:.1f} Hz<br>"
            f"  CCG suggestion: <b>{ccg_a}</b><br>"
            f"Unit B ({uid_b}): {lbl_b}  FR={fr_b:.1f} Hz<br>"
            f"  CCG suggestion: <b>{ccg_b}</b>"
        )

        # Set combo boxes to CCG suggestions
        self._set_combo(self._lbl_a, ccg_a if ccg_a and ccg_a != "unknown" else lbl_a)
        self._set_combo(self._lbl_b, ccg_b if ccg_b and ccg_b != "unknown" else lbl_b)

        # Store current pair context
        self._cur_pair_idx = pair_idx
        self._cur_uid_a = uid_a
        self._cur_uid_b = uid_b
        self._cur_idx_a = idx_a
        self._cur_idx_b = idx_b

        # Find position in _pair_order
        pos = np.where(self._pair_order == pair_idx)[0]
        self._cur_pair_pos = int(pos[0]) if len(pos) > 0 else -1

        # Draw plots
        self._draw_wf(self._wfA_plot, d, idx_a, f"WF Unit {uid_a} [{lbl_a}]")
        self._draw_wf(self._wfB_plot, d, idx_b, f"WF Unit {uid_b} [{lbl_b}]")
        self._draw_ccg(d, pair_idx, uid_a, uid_b, ptype, self._flip_ccg)
        self._draw_acg(self._acgA_plot, d, idx_a, uid_a, is_A=True)
        self._draw_acg(self._acgB_plot, d, idx_b, uid_b, is_A=False)

        # Select the row in the table
        self._select_table_row(pair_idx)

    def load_pair_for_unit(self, uid: int):
        """Open the strongest-scoring pair involving *uid*."""
        if self._data is None:
            return
        pairs = self._data.get_pairs_for_unit(uid)
        if len(pairs) == 0:
            return
        self._unit_pair_idxs = pairs
        self._unit_pair_pos = 0
        self.load_pair(int(pairs[0]))

    def _select_table_row(self, pair_idx: int):
        """Select the table row corresponding to pair_idx."""
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == pair_idx:
                self._table.blockSignals(True)
                self._table.selectRow(row)
                self._table.scrollToItem(item)
                self._table.blockSignals(False)
                break

    @staticmethod
    def _set_combo(cb: QComboBox, text: str):
        idx = cb.findText(text)
        if idx >= 0:
            cb.setCurrentIndex(idx)

    # ══════════════════════════════════════════════════════════════════════════
    #  Plot drawing
    # ══════════════════════════════════════════════════════════════════════════
    def _draw_wf(self, plot, data, unit_idx: int, title: str):
        plot.clear()
        plot.setTitle(title, color=FG, size="9pt")
        mean, std, _ = data.get_wf(unit_idx)
        n_ch, n_t = mean.shape

        # Normalise
        amp = float(mean[0].max() - mean[0].min())
        if amp == 0:
            amp = float(np.abs(mean).max()) or 1.0
        mn = mean / amp
        spacing = 1.5
        t = np.arange(n_t, dtype=np.float64)

        for ch in range(n_ch):
            color = WF_MAIN if ch == 0 else WF_OTHER
            width = 2.0 if ch == 0 else 1.0
            y_off = ch * spacing
            plot.plot(t, mn[ch] * self._wf_scale + y_off,
                      pen=pg.mkPen(color=color, width=width), antialias=True)

        plot.setXRange(0, n_t - 1, padding=0.02)
        plot.setYRange(-spacing, (n_ch - 1 + 1.5) * spacing, padding=0.02)

    def _draw_ccg(self, data, pair_idx: int, uid_a: int, uid_b: int,
                  ptype: str, flip: bool = False):
        self._ccg_plot.clear()
        self._ccg_bars = None

        ccg = data.get_pair_ccg(pair_idx)
        if flip:
            ccg = ccg[::-1]
        t = data.ccg_t_ms
        if len(t) == 0 or len(ccg) == 0:
            return

        bin_w = float(t[1] - t[0]) if len(t) > 1 else 0.1

        self._ccg_bars = pg.BarGraphItem(
            x=t, height=ccg, width=bin_w,
            brush=pg.mkBrush(CCG_COLOR), pen=pg.mkPen(None),
        )
        self._ccg_plot.addItem(self._ccg_bars)

        xlim = float(np.max(np.abs(t)))
        self._ccg_plot.setXRange(-xlim, xlim, padding=0.02)
        ymax = float(ccg.max()) * 1.12 + 0.5
        self._ccg_plot.setYRange(0, ymax, padding=0)
        self._ccg_plot.setTitle(
            f"CCG  {uid_a}\u2192{uid_b}  [{ptype}]", color=FG, size="9pt")
        self._ccg_plot.setLabel("bottom", "Lag (ms)", color=FG)

    def _redraw_wfs(self):
        """Redraw waveforms only (after wf_scale change)."""
        if not hasattr(self, "_cur_idx_a") or self._data is None:
            return
        d = self._data
        lbl_a = d.get_label(self._cur_idx_a)
        lbl_b = d.get_label(self._cur_idx_b)
        self._draw_wf(self._wfA_plot, d, self._cur_idx_a,
                      f"WF Unit {self._cur_uid_a} [{lbl_a}]")
        self._draw_wf(self._wfB_plot, d, self._cur_idx_b,
                      f"WF Unit {self._cur_uid_b} [{lbl_b}]")

    def _draw_acg(self, plot, data, unit_idx: int, uid: int, is_A: bool):
        plot.clear()

        acg, t_ms = data.get_acg_1d(unit_idx)
        xlim = 50.0
        mask = np.abs(t_ms) <= xlim
        t_z = t_ms[mask]
        a_z = acg[mask]
        bin_w = float(t_ms[1] - t_ms[0]) if len(t_ms) > 1 else 1.0

        bars = pg.BarGraphItem(
            x=t_z, height=a_z, width=bin_w,
            brush=pg.mkBrush(ACG_COLOR), pen=pg.mkPen(None),
        )
        plot.addItem(bars)

        side = "A" if is_A else "B"
        lbl = data.get_label(unit_idx)
        plot.setTitle(f"ACG Unit {side}: {uid} [{lbl}]", color=FG, size="9pt")
        plot.setXRange(-xlim, xlim, padding=0)
        plot.setYRange(0, float(a_z.max()) * 1.12 + 0.5, padding=0)

    # ══════════════════════════════════════════════════════════════════════════
    #  Navigation
    # ══════════════════════════════════════════════════════════════════════════
    def _on_table_select(self):
        row = self._table.currentRow()
        item = self._table.item(row, 0)
        if item is not None:
            pidx = item.data(Qt.UserRole)
            self.load_pair(pidx)

    def _go_prev(self):
        if self._cur_pair_pos > 0:
            self._cur_pair_pos -= 1
            self.load_pair(int(self._pair_order[self._cur_pair_pos]))

    def _go_next(self):
        if self._cur_pair_pos < len(self._pair_order) - 1:
            self._cur_pair_pos += 1
            self.load_pair(int(self._pair_order[self._cur_pair_pos]))

    def _go_next_unit_pair(self):
        """Next pair for the same unit (N key)."""
        if self._unit_pair_idxs is not None and self._unit_pair_pos < len(self._unit_pair_idxs) - 1:
            self._unit_pair_pos += 1
            self.load_pair(int(self._unit_pair_idxs[self._unit_pair_pos]))

    def _go_prev_unit_pair(self):
        """Previous pair for the same unit (P key)."""
        if self._unit_pair_idxs is not None and self._unit_pair_pos > 0:
            self._unit_pair_pos -= 1
            self.load_pair(int(self._unit_pair_idxs[self._unit_pair_pos]))

    # ══════════════════════════════════════════════════════════════════════════
    #  Accept / Reject
    # ══════════════════════════════════════════════════════════════════════════
    def _accept_pair(self):
        """Copy combo-box labels into data.labels for both units."""
        if self._data is None:
            return
        new_a = self._lbl_a.currentText()
        new_b = self._lbl_b.currentText()
        self._data.labels[self._cur_idx_a] = new_a
        self._data.labels[self._cur_idx_b] = new_b
        self.labels_changed.emit()
        # Advance to next pair
        self._go_next()

    def _reject_pair(self):
        """Set both units to 'unknown'."""
        if self._data is None:
            return
        self._data.labels[self._cur_idx_a] = "unknown"
        self._data.labels[self._cur_idx_b] = "unknown"
        self.labels_changed.emit()
        self._go_next()

    def _goto_unit_a(self):
        if hasattr(self, "_cur_idx_a"):
            self.go_to_unit.emit(self._cur_idx_a)

    def _goto_unit_b(self):
        if hasattr(self, "_cur_idx_b"):
            self.go_to_unit.emit(self._cur_idx_b)

    # ══════════════════════════════════════════════════════════════════════════
    #  Keyboard
    # ══════════════════════════════════════════════════════════════════════════
    def keyPressEvent(self, event):
        k = event.key()
        ctrl = event.modifiers() & Qt.ControlModifier
        if k == Qt.Key_Left:
            self._go_prev()
        elif k == Qt.Key_Right:
            self._go_next()
        elif k == Qt.Key_Up and ctrl:
            self._wf_scale = min(self._wf_scale * 1.5, 20.0)
            self._redraw_wfs()
        elif k == Qt.Key_Down and ctrl:
            self._wf_scale = max(self._wf_scale / 1.5, 0.05)
            self._redraw_wfs()
        elif k == Qt.Key_A:
            self._accept_pair()
        elif k == Qt.Key_R:
            self._reject_pair()
        elif k == Qt.Key_N:
            self._go_next_unit_pair()
        elif k == Qt.Key_P:
            self._go_prev_unit_pair()
        else:
            super().keyPressEvent(event)
