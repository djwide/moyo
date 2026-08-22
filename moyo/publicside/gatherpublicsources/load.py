"""Load gather-public-sources output into PublicSource objects.

Gather writes two artifacts under the project's ``public_sources/`` tree:

* ``sources.json`` — crawl (topic / token) results
* ``exploration.md`` — naive-prompt explore reports

Build Public Corpus extracts those into ``extracted.json`` (Kimi relevant
passages). Build Index and naive corpus compare read that extracted file,
not the raw gather dump.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Union

from shared_utils.ids import generate_id

from .schema import PublicSource, SourceType

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# File names gather actually writes (summary.json is crawl metadata, not a source).
CRAWL_SOURCES_NAME = "sources.json"
EXPLORE_REPORT_NAME = "exploration.md"
_SKIP_JSON_NAMES = {"summary.json", "extracted.json"}


def load_public_sources(root: PathLike) -> List[PublicSource]:
    """Recursively load crawl JSON and explore markdown under ``root``.

    ``root`` is typically ``projects/<slug>/public_sources/`` — the same
    directory Gather Public Sources writes to.
    """
    directory = Path(root)
    if not directory.exists():
        raise FileNotFoundError(f"Sources directory not found: {directory}")
    if directory.is_file():
        return list(_load_file(directory))

    sources: List[PublicSource] = []
    for path in _iter_gather_files(directory):
        sources.extend(_load_file(path))
    return sources


def _iter_gather_files(directory: Path) -> Iterable[Path]:
    crawl = sorted(directory.rglob(CRAWL_SOURCES_NAME))
    explore = sorted(directory.rglob(EXPLORE_REPORT_NAME))
    extra_json: List[Path] = []
    if not crawl:
        extra_json = [
            p
            for p in sorted(directory.rglob("*.json"))
            if p.name not in _SKIP_JSON_NAMES and p.name != CRAWL_SOURCES_NAME
        ]
    return [*crawl, *extra_json, *explore]


def _load_file(path: Path) -> List[PublicSource]:
    if path.name == EXPLORE_REPORT_NAME or path.suffix.lower() == ".md":
        return _load_exploration_md(path)
    if path.suffix.lower() == ".json":
        return _load_sources_json(path)
    return []


def _load_sources_json(path: Path) -> List[PublicSource]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping unreadable sources file %s: %s", path, exc)
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    sources: List[PublicSource] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        try:
            sources.append(PublicSource(**raw))
        except Exception as exc:
            logger.debug("Skipping invalid PublicSource in %s: %s", path, exc)
            continue
    return sources


def _load_exploration_md(path: Path) -> List[PublicSource]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Skipping unreadable exploration file %s: %s", path, exc)
        return []
    if not content.strip():
        return []
    title = path.parent.name if path.parent.name else path.stem
    return [
        PublicSource(
            id=generate_id("exploration"),
            title=title,
            content=content,
            source_type=SourceType.WEB_SEARCH,
            metadata={
                "path": str(path),
                "kind": EXPLORE_REPORT_NAME,
            },
            tags=["exploration"],
        )
    ]
