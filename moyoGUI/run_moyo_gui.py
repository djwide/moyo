#!/usr/bin/env python3
"""
Launcher script for moyo GUI (legacy entry point).

Prefer the installed console script instead:

    pip install -e .
    moyo-gui

This script is retained for convenience when running directly from a checkout
without an editable install.  It adds the repo root to sys.path so that the
`moyo` package (which includes `moyo.gui`) is importable even without
installation — no separate shared_utils path manipulation is needed because
shared_utils is vendored inside the `moyo` package.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

if __name__ == "__main__":
    from moyo.gui.app import main
    main()
