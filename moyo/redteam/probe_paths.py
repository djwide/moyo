"""Probe-path loading for hypothesis-driven blind probing.

A *probe path* is a curated list of secrets a particular kind of target would
find valuable to know. They live under the top-level ``probe_paths/`` directory,
one subdirectory per target customer, each containing one or more ``.txt`` files
with a single secret/topic per line (``#`` comments and blank lines ignored).

These functions resolve a probe path (by bundled name or filesystem path) and
load its entries as seed hypotheses for ``moyo-redteam blackbox``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def default_probe_paths_dir() -> Path:
    """Return the repository's top-level ``probe_paths/`` directory."""
    # moyo/redteam/probe_paths.py -> parents[2] is the repo root.
    return Path(__file__).resolve().parents[2] / "probe_paths"


def list_probe_paths() -> List[str]:
    """Return the names of bundled probe paths (subdirectories with .txt files)."""
    root = default_probe_paths_dir()
    if not root.is_dir():
        return []
    names = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(child.glob("*.txt")):
            names.append(child.name)
    return names


def resolve_probe_path(name_or_path: str) -> Path:
    """Resolve a probe path given a bundled name or a filesystem path.

    Resolution order:
      1. An existing file or directory at ``name_or_path``.
      2. A subdirectory of the bundled ``probe_paths/`` directory.
    """
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate

    bundled = default_probe_paths_dir() / name_or_path
    if bundled.exists():
        return bundled

    available = list_probe_paths()
    raise FileNotFoundError(
        f"Probe path not found: {name_or_path!r}. "
        f"Provide a path, or one of the bundled names: {', '.join(available) or 'none'}."
    )


def _read_seed_file(path: Path) -> List[str]:
    seeds: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            seeds.append(line)
    return seeds


def load_probe_seeds(name_or_path: str) -> List[str]:
    """Load seed topics from a probe path (a ``.txt`` file or a directory of them).

    Returns a de-duplicated, order-preserving list of seed strings.
    """
    path = resolve_probe_path(name_or_path)

    files: List[Path]
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
    else:
        files = [path]

    seeds: List[str] = []
    for f in files:
        seeds.extend(_read_seed_file(f))

    # De-duplicate while preserving order.
    seen = set()
    unique: List[str] = []
    for s in seeds:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique.append(s)

    logger.info("Loaded %d probe seeds from %s", len(unique), path)
    return unique
