"""
controls.py
-----------
Right-panel controls widget.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel,
    QDoubleSpinBox, QSpinBox, QHBoxLayout, QPushButton, QCheckBox,
)
from PyQt5.QtCore import pyqtSignal, Qt

_BTN = (
    "QPushButton { background: #1a1a3e; color: #ccccee; "
    "border: 1px solid #3a3a6a; border-radius: 3px; padding: 3px 10px; font-size: 9pt; }"
    "QPushButton:hover { background: #2a2a6a; }"
)
_GRP = (
    "QGroupBox { color: #8888cc; border: 1px solid #2a2a5a; "
    "border-radius: 4px; margin-top: 8px; padding: 4px; font-size: 8pt; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
)


def _dspin(mini, maxi, step, val, decimals=1) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(mini, maxi)
    sb.setSingleStep(step)
    sb.setValue(val)
    sb.setDecimals(decimals)
    sb.setFixedWidth(72)
    return sb


def _ispin(mini, maxi, step, val) -> QSpinBox:
    sb = QSpinBox()
    sb.setRange(mini, maxi)
    sb.setSingleStep(step)
    sb.setValue(val)
    sb.setFixedWidth(72)
    return sb


class ControlPanel(QWidget):

    wf_scale_changed    = pyqtSignal(float)
    wf_norm_changed     = pyqtSignal(bool)    # True = normalised
    wf_window_changed   = pyqtSignal(int)     # half-window in samples around peak
    acg_xlim_changed    = pyqtSignal(float)
    acg3d_clim_changed  = pyqtSignal(object)  # float | None
    acg3d_log_changed   = pyqtSignal(bool)    # True = log X axis
    acg3d_xlim_changed  = pyqtSignal(float)   # max lag to display (ms)
    nav_prev            = pyqtSignal()
    nav_next            = pyqtSignal()
    go_to_pair          = pyqtSignal()   # switch to Pairs tab for current unit

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(165)
        self.setMaximumWidth(230)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignTop)

        # ── Navigation ────────────────────────────────────────────────────────
        nav_grp = QGroupBox("Navigation")
        nav_grp.setStyleSheet(_GRP)
        nav_lay = QHBoxLayout(nav_grp)
        nav_lay.setSpacing(4)
        prev_btn = QPushButton("< Prev")
        next_btn = QPushButton("Next >")
        prev_btn.setStyleSheet(_BTN)
        next_btn.setStyleSheet(_BTN)
        prev_btn.clicked.connect(self.nav_prev)
        next_btn.clicked.connect(self.nav_next)
        nav_lay.addWidget(prev_btn)
        nav_lay.addWidget(next_btn)
        lay.addWidget(nav_grp)

        # ── Go to Pairs ─────────────────────────────────────────────────────
        pair_btn = QPushButton("Go to Pairs")
        pair_btn.setStyleSheet(_BTN)
        pair_btn.setToolTip("Switch to Pairs tab for this unit's CCG pairs")
        pair_btn.clicked.connect(self.go_to_pair)
        lay.addWidget(pair_btn)

        # ── Waveform ──────────────────────────────────────────────────────────
        wf_grp = QGroupBox("Waveform")
        wf_grp.setStyleSheet(_GRP)
        wf_lay = QVBoxLayout(wf_grp)
        wf_lay.setSpacing(4)

        # Normalized toggle
        self._wf_norm = QCheckBox("Normalised (peak-to-trough = 1)")
        self._wf_norm.setChecked(True)
        self._wf_norm.toggled.connect(self.wf_norm_changed)
        wf_lay.addWidget(self._wf_norm)

        # Scale
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Scale:"))
        self._wf_scale = _dspin(0.05, 20.0, 0.1, 1.0)
        self._wf_scale.valueChanged.connect(self.wf_scale_changed)
        row1.addWidget(self._wf_scale)
        wf_lay.addLayout(row1)

        # Window (half-width in samples around peak sample 40)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Window (+-samp):"))
        self._wf_win = _ispin(5, 40, 1, 40)   # default 40 = show all 81 samples
        self._wf_win.setToolTip(
            "Half-window in samples around the peak (max 40 = full 81-sample window)."
        )
        self._wf_win.valueChanged.connect(self.wf_window_changed)
        row2.addWidget(self._wf_win)
        wf_lay.addLayout(row2)

        lay.addWidget(wf_grp)

        # ── 1D ACG ────────────────────────────────────────────────────────────
        acg_grp = QGroupBox("1D ACG")
        acg_grp.setStyleSheet(_GRP)
        acg_lay = QHBoxLayout(acg_grp)
        acg_lay.addWidget(QLabel("+/- ms:"))
        self._acg_xlim = _dspin(2.0, 2000.0, 5.0, 50.0, decimals=0)
        self._acg_xlim.valueChanged.connect(self.acg_xlim_changed)
        acg_lay.addWidget(self._acg_xlim)
        lay.addWidget(acg_grp)

        # ── 3D ACG ────────────────────────────────────────────────────────────
        acg3d_grp = QGroupBox("3D ACG")
        acg3d_grp.setStyleSheet(_GRP)
        acg3d_lay = QVBoxLayout(acg3d_grp)
        acg3d_lay.setSpacing(4)

        # Log / linear X axis
        self._acg3d_log = QCheckBox("Log X axis")
        self._acg3d_log.setChecked(True)
        self._acg3d_log.toggled.connect(self.acg3d_log_changed)
        acg3d_lay.addWidget(self._acg3d_log)

        # X limit
        xlim_row = QHBoxLayout()
        xlim_row.addWidget(QLabel("+/- ms:"))
        self._acg3d_xlim = _dspin(1.0, 2000.0, 50.0, 200.0, decimals=0)
        self._acg3d_xlim.setToolTip("Max lag displayed on X axis (ms).")
        self._acg3d_xlim.valueChanged.connect(self.acg3d_xlim_changed)
        xlim_row.addWidget(self._acg3d_xlim)
        acg3d_lay.addLayout(xlim_row)

        # Colormap
        self._auto_clim = QCheckBox("Auto color")
        self._auto_clim.setChecked(True)
        self._auto_clim.toggled.connect(self._clim_toggled)
        acg3d_lay.addWidget(self._auto_clim)

        man_row = QHBoxLayout()
        man_row.addWidget(QLabel("Max:"))
        self._clim_max = _dspin(1.0, 5000.0, 10.0, 100.0, decimals=0)
        self._clim_max.setEnabled(False)
        self._clim_max.valueChanged.connect(self._clim_manual_changed)
        man_row.addWidget(self._clim_max)
        acg3d_lay.addLayout(man_row)

        lay.addWidget(acg3d_grp)
        lay.addStretch()

        # ── Unit info ─────────────────────────────────────────────────────────
        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #8888aa; font-size: 8pt;")
        self._info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(self._info_label)

    # ── Colormap ──────────────────────────────────────────────────────────────
    def _clim_toggled(self, auto: bool):
        self._clim_max.setEnabled(not auto)
        self.acg3d_clim_changed.emit(None if auto else self._clim_max.value())

    def _clim_manual_changed(self, val: float):
        if not self._auto_clim.isChecked():
            self.acg3d_clim_changed.emit(val)

    # ── Unit info ─────────────────────────────────────────────────────────────
    def set_unit_info(self, uid: int, label: str, depth: float,
                      fr: float, layer: str, c4: str, ccg_label: str = ""):
        lines = [
            f"<b>Unit {uid}</b>",
            f"Label : {label}",
            f"Depth : {depth:.0f} um",
            f"FR    : {fr:.1f} Hz",
        ]
        if ccg_label:
            lines.append(f"CCG   : {ccg_label}")
        if layer:
            lines.append(f"Layer : {layer}")
        if c4:
            lines.append(f"C4    : {c4}")
        self._info_label.setText("<br>".join(lines))
