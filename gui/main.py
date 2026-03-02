"""
main.py  --  Dataset Viewer entry point.

Usage
-----
    python -m cerebellum_cell_classifier.gui.main
    python -m cerebellum_cell_classifier.gui.main path/to/session_features.npz
"""

import os
import sys

sys.setrecursionlimit(5000)

# Ensure the project root is importable
_here = os.path.dirname(__file__)
_root = os.path.dirname(os.path.dirname(_here))
if _root not in sys.path:
    sys.path.insert(0, _root)


def main():
    from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
    from PyQt5.QtGui import QPalette, QColor

    app = QApplication(sys.argv)
    app.setApplicationName("Cerebellum Dataset Viewer")
    app.setStyle("Fusion")

    # Dark palette
    p = QPalette()
    p.setColor(QPalette.Window,          QColor(13,  13,  42))
    p.setColor(QPalette.WindowText,      QColor(224, 224, 224))
    p.setColor(QPalette.Base,            QColor( 9,   9,  30))
    p.setColor(QPalette.AlternateBase,   QColor(22,  22,  62))
    p.setColor(QPalette.Text,            QColor(224, 224, 224))
    p.setColor(QPalette.Button,          QColor(26,  26,  62))
    p.setColor(QPalette.ButtonText,      QColor(224, 224, 224))
    p.setColor(QPalette.Highlight,       QColor(58,  90, 154))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ToolTipBase,     QColor(13,  13,  42))
    p.setColor(QPalette.ToolTipText,     QColor(200, 200, 220))
    p.setColor(QPalette.Disabled, QPalette.Text,       QColor(100, 100, 140))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(100, 100, 140))
    app.setPalette(p)

    # pyqtgraph global config
    import pyqtgraph as pg
    pg.setConfigOption("background", "#16213e")
    pg.setConfigOption("foreground", "#e0e0e0")
    pg.setConfigOption("antialias",  True)

    # Resolve NPZ path
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        npz_path = sys.argv[1]
    else:
        npz_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open features file",
            os.path.expanduser("~"),
            "NPZ files (*_features.npz *.npz);;All files (*)",
        )
        if not npz_path:
            return

    try:
        from cerebellum_cell_classifier.gui.data_store import SessionData
        data = SessionData(npz_path)
        print(f"Loaded {data.session_name}: {data.n_units} units", flush=True)
    except Exception as e:
        QMessageBox.critical(None, "Load failed", str(e))
        raise

    from cerebellum_cell_classifier.gui.app_window import MainWindow
    win = MainWindow(data)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
