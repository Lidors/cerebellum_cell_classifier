"""
viewer.py  --  Dataset Viewer launcher.

Run from anywhere:
    python viewer.py
    python viewer.py path/to/session_features.npz
"""

import os
import sys

# Always add the project's parent dir to path so the package is importable
_project_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir  = os.path.dirname(_project_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from cerebellum_cell_classifier.gui.main import main

if __name__ == "__main__":
    main()
