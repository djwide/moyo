"""High-level ingestion pipeline orchestration for local and S3 storage."""

from __future__ import annotations

import io
import os
import tarfile
import zipfile
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from ..storage import Storage, get_storage, LocalStorage, safe_filename
from .validators import ValidationConfig, validate_file, ValidationError
from .loaders import load_text_from_bytes, LoadedDoc
from .normalize import build_normalized_doc
from .chunk import chunk_text, Chunk
from .manifest import ManifestWriter, ManifestRecord, ChunkRecord


@dataclass
class IngestConfig:
    max_bytes_per_file: int = 50 * 1024 * 1024
    max_pages_pdf: int = 500
    max_rows_xlsx: int = 200000
    max_expand_bytes_archive: int = 200 * 1024 * 1024
    max_files_archive: int = 5000
    ocr_enabled: bool = False
    chunk_strategy: str = "sentences"
    chunk_overlap: int = 1
    chunk_fixed_size: int = 1000
    allowed_mime: Optional[Sequence[str]] = None
    blocked_ext: Optional[Sequence[str]] = None

    def to_validation(self) -> ValidationConfig:
        from typing import Set
        return ValidationConfig(
            max_bytes_per_file=self.max_bytes_per_file,
            max_pages_pdf=self.max_pages_pdf,
            max_rows_xlsx=self.max_rows_xlsx,
            max_expand_bytes_archive=self.max_expand_bytes_archive,
            max_files_archive=self.max_files_archive,
            allowed_mime=set(self.allowed_mime) if self.allowed_mime else None,
            blocked_ext=set(self.blocked_ext) if self.blocked_ext else None,
        )


def _write_quarantine(store: Storage, base_dir: str, rel_key: str, error: Exception) -> None:
    safe_rel = rel_key.replace("/", "_").replace("\\", "_")
    key = os.path.join(base_dir, ".quarantine", f"{safe_rel}.log").replace("\\", "/")
    msg = f"{type(error).__name__}: {error}\n"
    store.write_bytes(key, msg.encode("utf-8"))


def _safe_rel_path(key: str) -> str:
    # sanitize key into a relative path safe for local filesystems
    key = key.replace("\\", "/").lstrip("/")
    parts = [safe_filename(p) for p in key.split("/") if p and p != "."]
    return "/".join(parts)


def _store_paths(base_dir: str, src_key: str) -> tuple[str, str, str]:
    # return paths under data/private/{raw,normalized,chunks}
    src_key = src_key.lstrip("/")
    rel = _safe_rel_path(src_key)
    raw_path = f"{base_dir}/raw/{rel}"
    norm_path = f"{base_dir}/normalized/{rel}.utf8"
    chunks_base = f"{base_dir}/chunks"
    return raw_path, norm_path, chunks_base


def _save_chunks(store: Storage, chunks_base: str, text_sha: str, chunks: List[Chunk]) -> List[ChunkRecord]:
    short = text_sha[:12]
    rels: List[ChunkRecord] = []
    for i, ch in enumerate(chunks):
        rel = f"{short}/chunk-{i+1:04d}.txt"
        key = f"{chunks_base}/{rel}"
        store.write_bytes(key, ch.text.encode("utf-8"))
        rels.append(ChunkRecord(id=ch.id, rel_path=f"{rel}", len=len(ch.text)))
    return rels


def _ingest_single(store: Storage, base_dir: str, src_key: str, data: bytes, cfg: IngestConfig, policy_tags: List[str], manifest: ManifestWriter) -> Optional[ManifestRecord]:
    vcfg = cfg.to_validation()
    try:
        mime = validate_file(src_key, data, vcfg)
        loaded = load_text_from_bytes(src_key, data, mime, vcfg)
        normalized = build_normalized_doc(loaded)

        # chunk
        chunks = chunk_text(
            normalized.text_norm,
            strategy=cfg.chunk_strategy,
            overlap=cfg.chunk_overlap,
            size=cfg.chunk_fixed_size,
        )

        # write outputs
        raw_key, norm_key, chunks_base = _store_paths(base_dir, src_key)
        store.write_bytes(raw_key, data)
        store.write_bytes(norm_key, normalized.text_norm.encode("utf-8"))
        chunk_recs = _save_chunks(store, chunks_base, normalized.text_sha256, chunks)

        rec = ManifestRecord(
            ingest_id=str(uuid4()),
            src_path=src_key,
            stored_at=norm_key,
            mime=normalized.mime,
            size_bytes=normalized.size_bytes,
            bytes_sha256=normalized.bytes_sha256,
            text_sha256=normalized.text_sha256,
            loader=str(loaded.meta.get("loader", "unknown")),
            normalized_at=now_iso(),
            language="und",
            policy_tags=policy_tags,
            notes="",
            chunks=chunk_recs,
        )
        written = manifest.append_if_new(rec)
        return rec if written else None
    except Exception as e:
        _write_quarantine(store, base_dir, src_key, e)
        return None


def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def uuid4() -> str:
    import uuid
    return str(uuid.uuid4())


def ingest_paths(paths: Sequence[str], *, policy_tags: List[str], cfg: IngestConfig, store: Storage, base_dir: str = "data/private") -> List[ManifestRecord]:
    """Ingest list of paths or storage keys into the given storage.

    Returns list of records written (skips duplicates by bytes_sha256).
    """
    manifest = ManifestWriter(manifest_key=f"{base_dir}/manifest.jsonl", storage=store)
    results: List[ManifestRecord] = []

    for path in paths:
        key = path
        # If local path and using LocalStorage, map into src_key mirror
        if isinstance(store, LocalStorage) and os.path.exists(path):
            # src_key mirrored under filename only; subdirs preserved
            root = os.path.commonprefix([os.getcwd(), os.path.abspath(path)])
            rel = os.path.basename(path)
            key = rel
            with open(path, "rb") as f:
                data = f.read()
        else:
            # read from storage directly
            data = store.read_bytes(key)

        rec = _ingest_single(store, base_dir, key, data, cfg, policy_tags, manifest)
        if rec is not None:
            results.append(rec)

        # Handle archives by iterating entries
        try:
            if key.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for zi in zf.infolist():
                        if zi.is_dir():
                            continue
                        entry_data = zf.read(zi.filename)
                        sub_key = f"{key}:{zi.filename}"
                        rec = _ingest_single(store, base_dir, sub_key, entry_data, cfg, policy_tags, manifest)
                        if rec is not None:
                            results.append(rec)
            elif key.lower().endswith((".tar", ".tgz", ".tar.gz")):
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
                    for ti in tf.getmembers():
                        if not ti.isfile():
                            continue
                        f = tf.extractfile(ti)
                        if not f:
                            continue
                        entry_data = f.read()
                        sub_key = f"{key}:{ti.name}"
                        rec = _ingest_single(store, base_dir, sub_key, entry_data, cfg, policy_tags, manifest)
                        if rec is not None:
                            results.append(rec)
        except Exception:
            # Any archive iteration failure will already be quarantined per entry
            pass

    return results


