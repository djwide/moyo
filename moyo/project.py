"""Per-engagement project folders for phrases, FAISS indexes, and public sources.

There is no global ``data/private/phrases`` store and no shared
``indexes/private`` bucket. Each project is a directory:

::

    projects/<slug>/
      phrases/                 pending.jsonl, corpus.jsonl, corpus.txt
      indexes/private/         private FAISS (one corpus per project)
      indexes/public/          public FAISS
      public_sources/          crawl / explore outputs
      compare/                 last Kimi naive-compare result

The GUI searches that tree for corpus files, ``*.faiss``, ``sources.json``,
and ``exploration.md``. Builds default to the conventional subfolders above.
Gather Public Sources writes ``sources.json`` (crawl) and ``exploration.md``
(explore) under ``public_sources/``; Build Public Corpus reads those files.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

_PHRASE_MARKERS = ("corpus.jsonl", "pending.jsonl")
_STATE_FILE = "project.json"


def workspace_root() -> Path:
    """Repo / workspace root (parent of the ``moyo`` package)."""
    return Path(__file__).resolve().parent.parent


def projects_root() -> Path:
    """Directory that contains project folders."""
    override = os.environ.get("MOYO_PROJECTS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    try:
        from moyo.config.settings import get_settings

        configured = getattr(get_settings(), "projects_dir", None)
        if configured:
            path = Path(configured)
            if not path.is_absolute():
                path = workspace_root() / path
            return path.resolve()
    except Exception:
        pass
    return (workspace_root() / "projects").resolve()


def slugify_project_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._-")
    return slug.lower() if slug else ""


def _is_project_dir(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith("."):
        return False
    markers = (
        path / "phrases",
        path / "indexes",
        path / "public_sources",
        path / "moyo-project.json",
    )
    if any(p.exists() for p in markers):
        return True
    return bool(find_phrase_dirs(path, max_depth=2) or find_faiss_files(path, max_depth=3))


@dataclass(frozen=True)
class MoyoProject:
    """One engagement: its own phrases corpus and FAISS indexes."""

    root: Path
    name: str

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "MoyoProject":
        root = Path(path).expanduser().resolve()
        return cls(root=root, name=root.name)

    @property
    def phrases_dir(self) -> Path:
        return self.root / "phrases"

    @property
    def private_index_dir(self) -> Path:
        return self.root / "indexes" / "private"

    @property
    def public_index_dir(self) -> Path:
        return self.root / "indexes" / "public"

    @property
    def public_sources_dir(self) -> Path:
        return self.root / "public_sources"

    @property
    def compare_dir(self) -> Path:
        return self.root / "compare"

    def ensure(self) -> "MoyoProject":
        for path in (
            self.root,
            self.phrases_dir,
            self.private_index_dir,
            self.public_index_dir,
            self.public_sources_dir,
            self.compare_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        meta = self.root / "moyo-project.json"
        if not meta.exists():
            meta.write_text(
                json.dumps(
                    {
                        "name": self.name,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return self

    def find_phrase_dirs(self) -> List[Path]:
        found = find_phrase_dirs(self.root)
        preferred = self.phrases_dir
        if preferred.exists() and preferred not in found:
            found.insert(0, preferred)
        elif preferred in found:
            found.remove(preferred)
            found.insert(0, preferred)
        return found

    def find_phrase_corpus(self) -> Optional[Path]:
        """Preferred approved-phrase file for FAISS builds."""
        conventional = self.phrases_dir / "corpus.jsonl"
        if conventional.is_file():
            return conventional
        for directory in self.find_phrase_dirs():
            corpus = directory / "corpus.jsonl"
            if corpus.is_file():
                return corpus
        matches = _rglob(self.root, "corpus.jsonl", max_depth=4)
        return matches[0] if matches else None

    def find_private_indexes(self) -> List[Path]:
        return _classify_faiss(self.root, self.private_index_dir, public=False)

    def find_public_indexes(self) -> List[Path]:
        return _classify_faiss(self.root, self.public_index_dir, public=True)

    def latest_private_index(self) -> Optional[Path]:
        found = self.find_private_indexes()
        return found[0] if found else (
            self.private_index_dir if self.private_index_dir.exists() else None
        )

    def latest_public_index(self) -> Optional[Path]:
        found = self.find_public_indexes()
        return found[0] if found else (
            self.public_index_dir if self.public_index_dir.exists() else None
        )

    def find_sources_dirs(self) -> List[Path]:
        hits = [p.parent for p in _rglob(self.root, "sources.json", max_depth=5)]
        hits += [p.parent for p in _rglob(self.root, "exploration.md", max_depth=5)]
        ordered: List[Path] = []
        for path in [self.public_sources_dir, *hits]:
            if path.exists() and path not in ordered:
                ordered.append(path)
        return ordered

    def find_explorations(self) -> List[Path]:
        return _rglob(self.root, "exploration.md", max_depth=5)

    def extracted_path(self) -> Path:
        """Canonical Kimi-extracted public corpus (``extracted.json``)."""
        from moyo.publicside.gatherpublicsources.extract import EXTRACTED_FILE_NAME

        return self.public_sources_dir / EXTRACTED_FILE_NAME

    def find_extracted(self) -> Optional[Path]:
        """Newest ``extracted.json`` under this project, conventional path first."""
        from moyo.publicside.gatherpublicsources.extract import EXTRACTED_FILE_NAME

        conventional = self.extracted_path()
        if conventional.is_file():
            return conventional
        hits = _rglob(self.root, EXTRACTED_FILE_NAME, max_depth=5)
        return hits[0] if hits else None


def create_project(name: str, *, root: Optional[Path] = None) -> MoyoProject:
    slug = slugify_project_name(name)
    if not slug:
        raise ValueError("Project name is empty")
    base = Path(root) if root is not None else projects_root()
    project = MoyoProject(root=(base / slug).resolve(), name=slug)
    return project.ensure()


def get_project(name: str, *, create: bool = False) -> MoyoProject:
    slug = slugify_project_name(name)
    if not slug:
        raise ValueError("Project name is empty")
    path = projects_root() / slug
    if path.exists():
        return MoyoProject.from_path(path)
    if create:
        return create_project(slug)
    raise FileNotFoundError(f"No project named {slug!r} under {projects_root()}")


def list_projects(*, extra: Sequence[Path] = ()) -> List[MoyoProject]:
    root = projects_root()
    found: dict[Path, MoyoProject] = {}
    if root.exists():
        for child in sorted(root.iterdir()):
            if _is_project_dir(child):
                proj = MoyoProject.from_path(child)
                found[proj.root] = proj
    for path in extra:
        try:
            proj = MoyoProject.from_path(path)
        except Exception:
            continue
        found[proj.root] = proj
    return sorted(found.values(), key=lambda p: p.name.lower())


def find_phrase_dirs(root: Path, *, max_depth: int = 4) -> List[Path]:
    """Directories under ``root`` that look like a PhraseStore."""
    root = Path(root)
    if not root.exists():
        return []
    dirs: List[Path] = []
    for marker in _PHRASE_MARKERS:
        for file_path in _rglob(root, marker, max_depth=max_depth):
            parent = file_path.parent
            if parent not in dirs:
                dirs.append(parent)
    return dirs


def find_faiss_files(root: Path, *, max_depth: int = 5) -> List[Path]:
    files = _rglob(root, "*.faiss", max_depth=max_depth)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def resolve_phrases_dir(
    *,
    project: Optional[str] = None,
    corpus_dir: Optional[Union[str, Path]] = None,
    create: bool = True,
) -> Path:
    """CLI/GUI helper: explicit corpus dir wins, else the named project's phrases/."""
    if corpus_dir:
        path = Path(corpus_dir)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    name = project or os.environ.get("MOYO_PROJECT") or _settings_project_name()
    if not name:
        raise ValueError(
            "No project selected. Pass --project NAME, set MOYO_PROJECT, "
            "or pass --corpus-dir. Phrases are per-project (not data/private/phrases)."
        )
    proj = get_project(name, create=create)
    if create:
        proj.ensure()
    return proj.phrases_dir


def resolve_private_index_dir(
    *,
    project: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    create: bool = True,
) -> Path:
    if output_dir:
        path = Path(output_dir)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    name = project or os.environ.get("MOYO_PROJECT") or _settings_project_name()
    if not name:
        raise ValueError(
            "No project selected. Pass --project NAME or set MOYO_PROJECT. "
            "FAISS indexes are per-project (not indexes/private)."
        )
    proj = get_project(name, create=create)
    if create:
        proj.ensure()
    return proj.private_index_dir


def load_saved_project() -> Optional[MoyoProject]:
    data = _read_state()
    raw = (data or {}).get("root") or (data or {}).get("name")
    if not raw:
        env = os.environ.get("MOYO_PROJECT") or _settings_project_name()
        if not env:
            return None
        path = Path(env)
        if path.exists() and path.is_dir() and (path.is_absolute() or "/" in env or "\\" in env):
            return MoyoProject.from_path(path)
        try:
            return get_project(env, create=False)
        except FileNotFoundError:
            return None
    path = Path(raw)
    if not path.is_absolute():
        try:
            return get_project(str(raw), create=False)
        except FileNotFoundError:
            path = projects_root() / slugify_project_name(str(raw))
    if path.exists():
        return MoyoProject.from_path(path)
    return None


def save_current_project(project: Optional[MoyoProject]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if project is None:
        if path.exists():
            path.unlink()
        return
    path.write_text(
        json.dumps({"name": project.name, "root": str(project.root)}, indent=2) + "\n",
        encoding="utf-8",
    )


def _settings_project_name() -> Optional[str]:
    try:
        from moyo.config.settings import get_settings

        value = getattr(get_settings(), "project", None)
        return str(value).strip() or None if value else None
    except Exception:
        return None


def _state_path() -> Path:
    override = os.environ.get("MOYO_CONFIG_DIR")
    base = Path(override) if override else workspace_root() / "config"
    return base / _STATE_FILE


def _read_state() -> Optional[dict]:
    path = _state_path()
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _rglob(root: Path, pattern: str, *, max_depth: int) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    matches: List[Path] = []
    root_depth = len(root.resolve().parts)
    for path in root.rglob(pattern):
        try:
            depth = len(path.resolve().parts) - root_depth
        except OSError:
            continue
        if depth <= max_depth and path.is_file():
            matches.append(path)
    matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return matches


def _classify_faiss(root: Path, conventional: Path, *, public: bool) -> List[Path]:
    """Return index *directories* (parent of .faiss), newest first."""
    files = find_faiss_files(root)
    dirs: List[Path] = []

    def _want(path: Path) -> bool:
        parts = {p.lower() for p in path.parts}
        if public:
            if "private" in parts and "public" not in parts:
                return False
            return "public" in parts or conventional in path.parents or path == conventional
        if "public" in parts and "private" not in parts:
            return False
        return "private" in parts or conventional in path.parents or path == conventional

    for faiss_file in files:
        directory = faiss_file.parent
        if not _want(directory):
            continue
        if directory not in dirs:
            dirs.append(directory)
    if conventional.exists() and conventional not in dirs:
        dirs.append(conventional)
    return dirs


def iter_project_names(projects: Iterable[MoyoProject]) -> List[str]:
    return [p.name for p in projects]
