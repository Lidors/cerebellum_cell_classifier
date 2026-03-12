"""
unit_table.py
-------------
Sortable, filterable unit list.

unit_selected(int)     -- emitted on row click (array index into SessionData)
label_changed(int,str) -- emitted when user changes a label via double-click
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QAbstractItemView,
    QLineEdit, QLabel, QHeaderView, QComboBox,
    QStyledItemDelegate, QStyleOptionViewItem,
)
from PyQt5.QtCore import pyqtSignal, Qt, QModelIndex
from PyQt5.QtGui import QColor, QFont


VALID_LABELS = ["PC", "CF", "MLI", "GC", "UBC", "MF", "GoC", "unknown"]

_LABEL_COLORS = {
    "PC":     "#1976D2",
    "PkC_ss": "#1976D2",
    "MLI":    "#E91E63",
    "GoC":    "#9C27B0",
    "GC":     "#4CAF50",
    "UBC":    "#FF9800",
    "MF":     "#F44336",
    "CF":     "#00BCD4",
    "PkC_cs": "#00BCD4",
}

_MFB_TIER_COLORS = {
    "core":     "#F44336",   # red — definite MFB
    "probable": "#FF9800",   # orange
}

_LABEL_COL   = 2   # column index of the editable Label cell
_CCG_LBL_COL = 3   # column index of the read-only CCG auto-label
_MFB_COL     = 4   # column index of the MFB tier


class _LabelDelegate(QStyledItemDelegate):
    """
    Shows a QComboBox dropdown when the user double-clicks a label cell.
    Emits label_committed(row, new_label) so the table widget can react.
    """
    label_committed = pyqtSignal(int, str)

    def createEditor(self, parent, option: QStyleOptionViewItem,
                     index: QModelIndex) -> QComboBox:
        cb = QComboBox(parent)
        cb.addItems(VALID_LABELS)
        current = index.data(Qt.DisplayRole) or ""
        if current in VALID_LABELS:
            cb.setCurrentText(current)
        return cb

    def setModelData(self, editor: QComboBox, model, index: QModelIndex):
        new_label = editor.currentText()
        model.setData(index, new_label, Qt.EditRole)
        self.label_committed.emit(index.row(), new_label)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class _NumItem(QTableWidgetItem):
    """Sort numerically when possible."""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return self.text() < other.text()


class UnitTableWidget(QWidget):
    unit_selected  = pyqtSignal(int)    # array index
    label_changed  = pyqtSignal(int, str)  # (array index, new label)

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self._filtered: list[int] = list(range(data.n_units))
        self._build_ui()
        self._populate()

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        frow = QHBoxLayout()
        frow.addWidget(QLabel("Filter:"))
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("label / layer / depth ...")
        self._filter.textChanged.connect(self._apply_filter)
        frow.addWidget(self._filter)
        lay.addLayout(frow)

        hint = QLabel("Double-click Label to edit")
        hint.setStyleSheet("color: #6666aa; font-size: 7pt;")
        lay.addWidget(hint)

        self._count_label = QLabel()
        lay.addWidget(self._count_label)

        cols = ["#", "ID", "Label", "CCG Label", "MFB Tier", "MFB Score", "Layer", "C4 pred", "Depth (um)", "FR (Hz)"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setDefaultSectionSize(20)
        self._table.setFont(QFont("", 9))
        self._table.itemSelectionChanged.connect(self._on_select)

        # Label-column delegate
        self._delegate = _LabelDelegate(self._table)
        self._delegate.label_committed.connect(self._on_label_committed)
        self._table.setItemDelegateForColumn(_LABEL_COL, self._delegate)

        # All other columns: read-only
        for col in range(self._table.columnCount()):
            if col != _LABEL_COL:
                self._table.setItemDelegateForColumn(col, _ReadOnlyDelegate(self._table))

        lay.addWidget(self._table)

    # ── Populate ────────────────────────────────────────────────────────────────
    def _populate(self, indices: list[int] | None = None):
        if indices is None:
            indices = list(range(self.data.n_units))
        self._filtered = indices

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(indices))

        for row, i in enumerate(indices):
            uid     = int(self.data.unit_ids[i])
            label   = self.data.get_label(i)
            ccg_lbl = self.data.get_ccg_label(i)
            mfb_tier  = self.data.get_mfb_tier(i)
            mfb_score = self.data.get_mfb_score(i)
            layer   = self.data.get_layer(i)
            c4      = self.data.get_c4_pred(i)
            depth   = self.data.get_depth(i)
            fr      = self.data.get_mean_fr(i)

            mfb_score_str = f"{mfb_score:.3f}" if mfb_score == mfb_score else ""  # NaN check

            cells = [
                _NumItem(str(i)),
                _NumItem(str(uid)),
                QTableWidgetItem(label),      # label — editable via delegate
                QTableWidgetItem(ccg_lbl),    # CCG auto-label — read-only
                QTableWidgetItem(mfb_tier),   # MFB tier
                _NumItem(mfb_score_str),      # MFB score
                QTableWidgetItem(layer),
                QTableWidgetItem(c4),
                _NumItem(f"{depth:.0f}"),
                _NumItem(f"{fr:.1f}"),
            ]
            for col, item in enumerate(cells):
                item.setTextAlignment(Qt.AlignCenter)
                item.setData(Qt.UserRole, i)   # always store array index
                if col == _LABEL_COL:
                    color = _LABEL_COLORS.get(label)
                    if color:
                        item.setForeground(QColor(color))
                if col == _CCG_LBL_COL:
                    color = _LABEL_COLORS.get(ccg_lbl)
                    if color:
                        item.setForeground(QColor(color))
                if col == _MFB_COL:
                    color = _MFB_TIER_COLORS.get(mfb_tier)
                    if color:
                        item.setForeground(QColor(color))
                self._table.setItem(row, col, item)

        self._table.setSortingEnabled(True)
        self._count_label.setText(f"{len(indices)} / {self.data.n_units} units")

    # ── Filter ──────────────────────────────────────────────────────────────────
    def _apply_filter(self, text: str):
        text = text.strip().lower()
        if not text:
            self._populate()
            return
        matched = [
            i for i in range(self.data.n_units)
            if (text in self.data.get_label(i).lower()
                or text in self.data.get_ccg_label(i).lower()
                or text in self.data.get_mfb_tier(i).lower()
                or text in self.data.get_layer(i).lower()
                or text in self.data.get_c4_pred(i).lower()
                or text in str(self.data.unit_ids[i]))
        ]
        self._populate(matched)

    # ── Selection ────────────────────────────────────────────────────────────────
    def _on_select(self):
        if getattr(self, "_in_select", False):
            return
        self._in_select = True
        try:
            row  = self._table.currentRow()
            item = self._table.item(row, 0)
            if item is not None:
                self.unit_selected.emit(item.data(Qt.UserRole))
        finally:
            self._in_select = False

    # ── Label edit ───────────────────────────────────────────────────────────────
    def _on_label_committed(self, row: int, new_label: str):
        """Called when the delegate commits a new label value."""
        # Get array index from the row-index cell (col 0), which may be sorted
        item0 = self._table.item(row, 0)
        if item0 is None:
            return
        arr_i = item0.data(Qt.UserRole)

        # Update label cell colour
        label_item = self._table.item(row, _LABEL_COL)
        if label_item is not None:
            color = _LABEL_COLORS.get(new_label)
            if color:
                label_item.setForeground(QColor(color))
            else:
                label_item.setForeground(QColor(FG_DEFAULT))

        # Update data store in-memory
        self.data.labels[arr_i] = new_label
        self.label_changed.emit(arr_i, new_label)

    def refresh_unit_row(self, unit_i: int):
        """Update label and MFB cells for a single unit without rebuilding the table."""
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is None or item.data(Qt.UserRole) != unit_i:
                continue
            label    = self.data.get_label(unit_i)
            mfb_tier = self.data.get_mfb_tier(unit_i)

            label_item = self._table.item(row, _LABEL_COL)
            if label_item:
                label_item.setText(label)
                color = _LABEL_COLORS.get(label)
                label_item.setForeground(QColor(color) if color else QColor(FG_DEFAULT))

            mfb_item = self._table.item(row, _MFB_COL)
            if mfb_item:
                mfb_item.setText(mfb_tier)
                color = _MFB_TIER_COLORS.get(mfb_tier)
                mfb_item.setForeground(QColor(color) if color else QColor(FG_DEFAULT))
            break

    def get_unit_at_offset(self, unit_i: int, delta: int) -> int:
        """Return the unit array-index that is *delta* rows away from *unit_i*
        in the current (sorted/filtered) table view.  Returns -1 if out of range."""
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == unit_i:
                new_row = row + delta
                if 0 <= new_row < self._table.rowCount():
                    new_item = self._table.item(new_row, 0)
                    if new_item is not None:
                        return new_item.data(Qt.UserRole)
                return -1
        return -1

    def select_row_for_index(self, unit_i: int):
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == unit_i:
                self._table.blockSignals(True)
                self._table.selectRow(row)
                self._table.blockSignals(False)
                self._table.scrollToItem(item)
                break


FG_DEFAULT = "#e0e0e0"


class _ReadOnlyDelegate(QStyledItemDelegate):
    """Prevents editing on non-label columns."""
    def createEditor(self, parent, option, index):
        return None
