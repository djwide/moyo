#!/usr/bin/env python3
"""
Launcher script for moyo GUI.

Prefer running after `pip install -e .` from the repo root; this script adds
the project root and the vendored shared_utils directory to sys.path as a
fallback for development without an editable install.
"""

import sys
from pathlib import Path

# Project root is one level above this file (repo root).
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# shared_utils is vendored inside the repo root, not in a sibling directory.
shared_utils_path = project_root / "shared_utils"
if shared_utils_path.is_dir():
    sys.path.insert(0, str(shared_utils_path))

if __name__ == "__main__":
    from moyo_gui import main
    main()
