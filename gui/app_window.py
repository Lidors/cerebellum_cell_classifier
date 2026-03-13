"""
app_window.py
-------------
Main application window with:
  - Session tab bar (multi-session, closeable)
  - Sub-tabs: Units view | Pairs view
  - Units: Left panel (unit table), Centre (WF + 1D ACG + 3D ACG), Right (controls)
  - Pairs: Pair explorer with CCG / WF / ACG plots

Keyboard shortcuts
------------------
  Left / Right   -- previous / next unit (Units) or pair (Pairs)
  Ctrl+O         -- add session
  Ctrl+Q         -- quit
  Tab            -- switch between Units / Pairs views
  A / R          -- accept / reject pair (Pairs view)
  N / P          -- next / prev pair for same unit (Pairs view)
  M              -- confirm MFB (label→MF)             (Units view)
  Shift+M        -- reject MFB (label→unknown)         (Units view)
  F              -- confirm CF (label→CF)              (Units view)
  Shift+F        -- reject CF (label→unknown)          (Units view)
"""

from __future__ import annotations

import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QTabBar, QTabWidget,
    QPushButton, QFileDialog, QMessageBox, QStatusBar, QAction, QShortcut,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence

from cerebellum_cell_classifier.gui.data_store  import SessionData
from cerebellum_cell_classifier.gui.unit_table  import UnitTableWidget
from cerebellum_cell_classifier.gui.plots_panel import PlotsPanel
from cerebellum_cell_classifier.gui.controls    import ControlPanel
from cerebellum_cell_classifier.gui.pair_panel  import PairPanel

DARK = (
    "QMainWindow, QSplitter { background: #0d0d2a; }"
    "QMenuBar { background: #111128; color: #ccccee; }"
    "QMenuBar::item:selected { background: #2a2a6a; }"
    "QMenu { background: #111128; color: #ccccee; border: 1px solid #3a3a6a; }"
    "QMenu::item:selected { background: #2a2a6a; }"
    "QStatusBar { background: #111128; color: #8888aa; font-size: 8pt; }"
    "QSplitter::handle { background: #2a2a5a; width: 3px; }"
)

_TABBAR = (
    "QTabBar { background: #0d0d2a; }"
    "QTabBar::tab {"
    "  background: #111128; color: #6666aa;"
    "  padding: 3px 16px 3px 12px;"
    "  border: 1px solid #2a2a5a; border-bottom: none;"
    "  border-radius: 3px 3px 0 0; margin-right: 2px; font-size: 9pt; }"
    "QTabBar::tab:selected { background: #1a1a4a; color: #ccccff; }"
    "QTabBar::tab:hover    { background: #161640; color: #aaaaee; }"
)

_SUBTAB = (
    "QTabWidget::pane { border: none; background: #0d0d2a; }"
    "QTabBar { background: #0d0d2a; }"
    "QTabBar::tab {"
    "  background: #111128; color: #6666aa;"
    "  padding: 2px 14px;"
    "  border: 1px solid #2a2a5a; border-bottom: none;"
    "  border-radius: 3px 3px 0 0; margin-right: 2px; font-size: 8pt; }"
    "QTabBar::tab:selected { background: #1a1a4a; color: #ccccff; }"
    "QTabBar::tab:hover    { background: #161640; color: #aaaaee; }"
)


class MainWindow(QMainWindow):
    def __init__(self, initial_data: SessionData):
        super().__init__()
        self.setStyleSheet(DARK)
        self.setWindowTitle("Cerebellum Cell Classifier — Dataset Viewer")
        self.resize(1400, 860)

        self._sessions: list[tuple[str, SessionData]] = [
            (initial_data.session_name, initial_data)
        ]
        self._cur_sess = 0
        self._unit_i   = 0

        # Debounce rapid navigation (arrow keys held)
        self._pending_i = 0
        self._nav_timer = QTimer(self)
        self._nav_timer.setSingleShot(True)
        self._nav_timer.setInterval(60)
        self._nav_timer.timeout.connect(self._do_load_unit)

        self._build_ui()
        self._connect_signals()
        self._build_global_shortcuts()
        self._load_unit(0)

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Session tab bar row
        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(4, 2, 4, 0)
        tab_row.setSpacing(4)

        self._tabs = QTabBar()
        self._tabs.setStyleSheet(_TABBAR)
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(False)
        self._tabs.setExpanding(False)
        self._tabs.addTab(self._sessions[0][0])
        self._tabs.currentChanged.connect(self._switch_session)
        self._tabs.tabCloseRequested.connect(self._close_tab)
        tab_row.addWidget(self._tabs)

        add_btn = QPushButton("+")
        add_btn.setFixedSize(24, 22)
        add_btn.setToolTip("Add session (Ctrl+O)")
        add_btn.setStyleSheet(
            "QPushButton { background: #1a1a3e; color: #8888cc; "
            "border: 1px solid #3a3a6a; border-radius: 3px; font-size: 12pt; }"
            "QPushButton:hover { background: #2a2a6a; color: #ccccff; }"
        )
        add_btn.clicked.connect(self._add_session)
        tab_row.addWidget(add_btn)
        tab_row.addStretch()
        root.addLayout(tab_row)

        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #2a2a5a;")
        root.addWidget(sep)

        # Sub-tab widget: Units | Pairs
        self._view_tabs = QTabWidget()
        self._view_tabs.setStyleSheet(_SUBTAB)
        root.addWidget(self._view_tabs, stretch=1)

        # --- Units tab ---
        units_w = QWidget()
        units_lay = QVBoxLayout(units_w)
        units_lay.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        units_lay.addWidget(splitter)

        self._unit_table = UnitTableWidget(self._sessions[0][1])
        self._plots = PlotsPanel()
        self._controls = ControlPanel()

        splitter.addWidget(self._unit_table)
        splitter.addWidget(self._plots)
        splitter.addWidget(self._controls)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 900, 200])

        self._view_tabs.addTab(units_w, "Units")

        # --- Pairs tab ---
        self._pair_panel = PairPanel()
        self._pair_panel.set_data(self._sessions[0][1])
        self._view_tabs.addTab(self._pair_panel, "Pairs")

        self._build_menu()
        self.setStatusBar(QStatusBar())

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")

        add_act = QAction("&Add session...", self)
        add_act.setShortcut(QKeySequence("Ctrl+O"))
        add_act.triggered.connect(self._add_session)
        fm.addAction(add_act)

        save_act = QAction("&Save labels to CSV...", self)
        save_act.setShortcut(QKeySequence("Ctrl+S"))
        save_act.triggered.connect(self._save_labels)
        fm.addAction(save_act)

        fm.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        fm.addAction(quit_act)

        hm = mb.addMenu("&Help")
        kb_act = QAction("Keyboard shortcuts", self)
        kb_act.triggered.connect(self._show_shortcuts)
        hm.addAction(kb_act)

    # ── Signals ─────────────────────────────────────────────────────────────────
    def _connect_signals(self):
        self._unit_table.unit_selected.connect(self._load_unit)
        self._unit_table.label_changed.connect(self._on_label_changed)
        c = self._controls
        c.wf_scale_changed.connect(self._on_wf_scale)
        c.wf_norm_changed.connect(self._on_wf_norm)
        c.wf_hp_changed.connect(self._on_wf_hp)
        c.wf_window_changed.connect(self._on_wf_window)
        c.acg_xlim_changed.connect(self._on_acg_xlim)
        c.acg3d_clim_changed.connect(self._on_acg3d_clim)
        c.acg3d_log_changed.connect(self._on_acg3d_log)
        c.acg3d_xlim_changed.connect(self._on_acg3d_xlim)
        c.nav_prev.connect(self._prev)
        c.nav_next.connect(self._next)
        c.go_to_pair.connect(self._switch_to_pairs)
        c.mfb_confirm.connect(self._confirm_mfb)
        c.mfb_reject.connect(self._reject_mfb)
        c.cf_confirm.connect(self._confirm_cf)
        c.cf_reject.connect(self._reject_cf)

        # Pair panel
        self._pair_panel.go_to_unit.connect(self._pair_goto_unit)
        self._pair_panel.labels_changed.connect(self._on_pair_labels_changed)

    def _build_global_shortcuts(self):
        """Shortcuts that fire regardless of which widget has keyboard focus."""
        # Units view — MFB
        sc_m = QShortcut(QKeySequence("M"), self)
        sc_m.setContext(Qt.ApplicationShortcut)
        sc_m.activated.connect(self._confirm_mfb)

        sc_sm = QShortcut(QKeySequence("Shift+M"), self)
        sc_sm.setContext(Qt.ApplicationShortcut)
        sc_sm.activated.connect(self._reject_mfb)

        # Units view — CF
        sc_f = QShortcut(QKeySequence("F"), self)
        sc_f.setContext(Qt.ApplicationShortcut)
        sc_f.activated.connect(self._confirm_cf)

        sc_sf = QShortcut(QKeySequence("Shift+F"), self)
        sc_sf.setContext(Qt.ApplicationShortcut)
        sc_sf.activated.connect(self._reject_cf)

        # Pairs view — accept / reject / navigate
        sc_a = QShortcut(QKeySequence("A"), self)
        sc_a.setContext(Qt.ApplicationShortcut)
        sc_a.activated.connect(self._pairs_accept)

        sc_r = QShortcut(QKeySequence("R"), self)
        sc_r.setContext(Qt.ApplicationShortcut)
        sc_r.activated.connect(self._pairs_reject)

        sc_n = QShortcut(QKeySequence("N"), self)
        sc_n.setContext(Qt.ApplicationShortcut)
        sc_n.activated.connect(self._pairs_next_unit)

        sc_p = QShortcut(QKeySequence("P"), self)
        sc_p.setContext(Qt.ApplicationShortcut)
        sc_p.activated.connect(self._pairs_prev_unit)

    # ── Session management ───────────────────────────────────────────────────────
    def _data(self) -> SessionData:
        return self._sessions[self._cur_sess][1]

    def _add_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open features file", "",
            "NPZ files (*_features.npz *.npz);;All files (*)"
        )
        if not path:
            return
        try:
            new_data = SessionData(path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        name = new_data.session_name or os.path.splitext(os.path.basename(path))[0]
        self._sessions.append((name, new_data))
        self._tabs.blockSignals(True)
        idx = self._tabs.addTab(name)
        self._tabs.blockSignals(False)
        self._tabs.setCurrentIndex(idx)

    def _switch_session(self, idx: int):
        if idx < 0 or idx >= len(self._sessions):
            return
        self._cur_sess = idx
        data = self._data()
        self._unit_table.data = data
        self._unit_table._populate()
        self._pair_panel.set_data(data)
        # Reset unit to 0 in new session
        self._unit_i = 0
        self._load_unit(0)

    def _close_tab(self, idx: int):
        if len(self._sessions) <= 1:
            return
        self._sessions.pop(idx)
        self._tabs.blockSignals(True)
        self._tabs.removeTab(idx)
        new_idx = min(self._cur_sess, len(self._sessions) - 1)
        self._tabs.setCurrentIndex(new_idx)
        self._tabs.blockSignals(False)
        self._switch_session(new_idx)

    # ── Unit loading ─────────────────────────────────────────────────────────────
    def _load_unit(self, i: int):
        """Schedule unit load; coalesces rapid navigation calls."""
        self._pending_i = i
        self._nav_timer.start()

    def _do_load_unit(self):
        i = self._pending_i
        data = self._data()
        if i < 0 or i >= data.n_units:
            return
        self._unit_i = i
        self._plots.update_unit(i, data)
        self._unit_table.select_row_for_index(i)

        uid   = int(data.unit_ids[i])
        label = data.get_label(i)
        depth = data.get_depth(i)
        fr    = data.get_mean_fr(i)
        layer = data.get_layer(i)
        c4    = data.get_c4_pred(i)
        ccg_lbl  = data.get_ccg_label(i)
        mfb_tier = data.get_mfb_tier(i)

        self._controls.set_unit_info(uid, label, depth, fr, layer, c4, ccg_lbl, mfb_tier)
        sess = self._sessions[self._cur_sess][0]
        self.statusBar().showMessage(
            f"[{sess}]  Unit {uid}  |  {label}  |  {depth:.0f} um  "
            f"|  {fr:.1f} Hz  |  {i+1}/{data.n_units}"
        )

    # ── Navigation ────────────────────────────────────────────────────────────────
    def _prev(self):
        new_i = self._unit_table.get_unit_at_offset(self._unit_i, -1)
        if new_i >= 0:
            self._load_unit(new_i)

    def _next(self):
        new_i = self._unit_table.get_unit_at_offset(self._unit_i, +1)
        if new_i >= 0:
            self._load_unit(new_i)

    # ── Control responses ─────────────────────────────────────────────────────────
    def _on_wf_scale(self, v: float):
        self._plots.set_wf_scale(v)
        self._do_load_unit()

    def _on_wf_norm(self, norm: bool):
        self._plots.set_wf_norm(norm)
        self._do_load_unit()

    def _on_wf_hp(self, hp: bool):
        self._plots.set_wf_hp(hp)
        self._do_load_unit()

    def _on_wf_window(self, hw: int):
        self._plots.set_wf_half_win(hw)
        self._do_load_unit()

    def _on_acg_xlim(self, v: float):
        self._plots.set_acg_xlim(v)
        self._do_load_unit()

    def _on_acg3d_clim(self, v):
        self._plots.set_acg3d_clim(v)
        self._do_load_unit()

    def _on_acg3d_log(self, log: bool):
        self._plots.set_acg3d_log(log)
        self._do_load_unit()

    def _on_acg3d_xlim(self, v: float):
        self._plots.set_acg3d_xlim(v)
        self._do_load_unit()

    def _on_label_changed(self, arr_i: int, new_label: str):
        """Label edited in table -- refresh status bar."""
        if arr_i == self._unit_i:
            data = self._data()
            uid  = int(data.unit_ids[arr_i])
            self.statusBar().showMessage(
                f"Unit {uid} label -> {new_label}", 3000
            )

    # ── Pair ↔ Unit view switching ────────────────────────────────────────────
    def _switch_to_pairs(self):
        """Switch to Pairs tab, showing pairs for the current unit."""
        data = self._data()
        uid = int(data.unit_ids[self._unit_i])
        self._view_tabs.setCurrentIndex(1)   # Pairs tab
        self._pair_panel.load_pair_for_unit(uid)

    def _pair_goto_unit(self, unit_idx: int):
        """PairPanel asked to go back to Units view for a specific unit."""
        self._view_tabs.setCurrentIndex(0)   # Units tab
        self._load_unit(unit_idx)

    def _on_pair_labels_changed(self):
        """Labels were changed in pair view — refresh unit table."""
        self._unit_table._populate()
        # Re-select the current unit row
        self._unit_table.select_row_for_index(self._unit_i)

    def _is_pairs_view(self) -> bool:
        return self._view_tabs.currentIndex() == 1

    # ── Keys ─────────────────────────────────────────────────────────────────────
    _WF_SCALE_STEP = 0.15   # multiplicative step for Ctrl+Up/Down

    def keyPressEvent(self, event):
        k   = event.key()
        mod = event.modifiers()

        # Tab key: toggle between Units and Pairs views
        if k == Qt.Key_Tab and not (mod & Qt.ControlModifier):
            new_idx = 1 - self._view_tabs.currentIndex()
            self._view_tabs.setCurrentIndex(new_idx)
            return

        # Delegate to pair panel when in Pairs view
        if self._is_pairs_view():
            self._pair_panel.keyPressEvent(event)
            return

        # Units view keys
        if k == Qt.Key_Left:
            self._prev()
        elif k == Qt.Key_Right:
            self._next()
        elif mod & Qt.ControlModifier and k == Qt.Key_Up:
            self._scale_wf(+1)
        elif mod & Qt.ControlModifier and k == Qt.Key_Down:
            self._scale_wf(-1)
        else:
            super().keyPressEvent(event)

    def _scale_wf(self, direction: int):
        """Increase (+1) or decrease (-1) WF scale by one step."""
        cur = self._controls._wf_scale.value()
        if direction > 0:
            new = cur * (1 + self._WF_SCALE_STEP)
        else:
            new = cur / (1 + self._WF_SCALE_STEP)
        new = max(0.05, round(new, 2))
        self._controls._wf_scale.setValue(new)
        # valueChanged signal fires automatically → _on_wf_scale → _do_load_unit

    # ── MFB confirm / reject ─────────────────────────────────────────────────────
    def _confirm_mfb(self):
        """Set label → MF for current unit and auto-save to npz."""
        if self._is_pairs_view():
            return
        data = self._data()
        i = self._unit_i
        data.labels[i] = "MF"
        self._unit_table.refresh_unit_row(i)
        self._do_load_unit()
        data.save_labels_to_npz()
        uid = int(data.unit_ids[i])
        self.statusBar().showMessage(f"Unit {uid}: label → MF  (saved)", 3000)

    def _reject_mfb(self):
        """Set label → unknown for current unit and auto-save to npz."""
        if self._is_pairs_view():
            return
        data = self._data()
        i = self._unit_i
        data.labels[i] = "unknown"
        self._unit_table.refresh_unit_row(i)
        self._do_load_unit()
        data.save_labels_to_npz()
        uid = int(data.unit_ids[i])
        self.statusBar().showMessage(f"Unit {uid}: label → unknown  (saved)", 3000)

    # ── CF confirm / reject ───────────────────────────────────────────────────
    def _confirm_cf(self):
        """Set label → CF for current unit and auto-save to npz."""
        if self._is_pairs_view():
            return
        data = self._data()
        i = self._unit_i
        data.labels[i] = "CF"
        self._unit_table.refresh_unit_row(i)
        self._do_load_unit()
        data.save_labels_to_npz()
        uid = int(data.unit_ids[i])
        self.statusBar().showMessage(f"Unit {uid}: label → CF  (saved)", 3000)

    def _reject_cf(self):
        """Set label → unknown for current unit and auto-save to npz."""
        if self._is_pairs_view():
            return
        data = self._data()
        i = self._unit_i
        data.labels[i] = "unknown"
        self._unit_table.refresh_unit_row(i)
        self._do_load_unit()
        data.save_labels_to_npz()
        uid = int(data.unit_ids[i])
        self.statusBar().showMessage(f"Unit {uid}: label → unknown  (saved)", 3000)

    # ── Pairs view wrappers ───────────────────────────────────────────────────
    def _pairs_accept(self):
        if self._is_pairs_view():
            self._pair_panel._accept_pair()

    def _pairs_reject(self):
        if self._is_pairs_view():
            self._pair_panel._reject_pair()

    def _pairs_next_unit(self):
        if self._is_pairs_view():
            self._pair_panel._go_next_unit_pair()

    def _pairs_prev_unit(self):
        if self._is_pairs_view():
            self._pair_panel._go_prev_unit_pair()

    # ── Save labels ──────────────────────────────────────────────────────────────
    def _save_labels(self):
        import pandas as pd
        data = self._data()
        csv_path = data._npz_path.parent / f"{data.session_name}_labels.csv"
        cols = {
            "unit_id": data.unit_ids,
            "label":   data.labels,
        }
        if data.has_ccg:
            cols["ccg_auto_label"] = data.ccg_auto_labels
        pd.DataFrame(cols).to_csv(csv_path, index=False)
        self.statusBar().showMessage(f"Labels saved to {csv_path.name}", 4000)

    # ── Help ─────────────────────────────────────────────────────────────────────
    def _show_shortcuts(self):
        QMessageBox.information(self, "Keyboard shortcuts",
            "<b>Views</b><br>"
            "&nbsp; Tab  &mdash; toggle between Units / Pairs views<br><br>"
            "<b>Units view — Navigation</b><br>"
            "&nbsp; Left / Right arrow  &mdash; previous / next unit<br>"
            "&nbsp; Prev / Next buttons in control panel<br><br>"
            "<b>Units view — Waveform scale</b><br>"
            "&nbsp; Ctrl + Up    &mdash; increase scale<br>"
            "&nbsp; Ctrl + Down  &mdash; decrease scale<br><br>"
            "<b>Pairs view</b><br>"
            "&nbsp; Left / Right  &mdash; previous / next pair<br>"
            "&nbsp; A  &mdash; accept CCG labels<br>"
            "&nbsp; R  &mdash; reject (set to unknown)<br>"
            "&nbsp; N / P  &mdash; next / previous pair for same unit<br><br>"
            "<b>MFB (Units view)</b><br>"
            "&nbsp; M        &mdash; set label → MF<br>"
            "&nbsp; Shift+M  &mdash; set label → unknown<br><br>"
            "<b>CF (Units view)</b><br>"
            "&nbsp; F        &mdash; set label → CF<br>"
            "&nbsp; Shift+F  &mdash; set label → unknown<br><br>"
            "<b>Labels</b><br>"
            "&nbsp; Double-click Label cell in table &mdash; edit label<br>"
            "&nbsp; Ctrl+S  &mdash; save labels to CSV (auto-saved to session folder)<br><br>"
            "<b>Sessions</b><br>"
            "&nbsp; Ctrl+O  &mdash; add session<br>"
            "&nbsp; Click tab to switch, x to close<br><br>"
            "<b>File</b><br>"
            "&nbsp; Ctrl+Q  &mdash; quit<br>"
        )
