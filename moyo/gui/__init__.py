"""moyo desktop GUI package.

Submodules such as ``cloud_compute`` must import without PyQt5. ``main`` is
loaded lazily so ``from moyo.gui import main`` still works when the GUI extra
is installed.
"""

from typing import Any

__all__ = ["main"]


def __getattr__(name: str) -> Any:
    if name == "main":
        from .app import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
