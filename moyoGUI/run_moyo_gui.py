#!/usr/bin/env python3
"""
Launcher script for Moyo GUI
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add shared_utils to the path
shared_utils_path = project_root.parent / "shared_utils"
sys.path.insert(0, str(shared_utils_path))

if __name__ == "__main__":
    from moyo_gui import main
    main()
