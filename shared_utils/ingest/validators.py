"""Validators for ingestion: MIME, size, archive guard, allow/block lists."""

from __future__ import annotations

import io
import os
import tarfile
import zipfile
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set


def _sniff_mime(data: bytes, fallback_name: Optional[str] = None) -> str:
    try:
        import magic  # python-magic
        m = magic.Magic(mime=True)
        return m.from_buffer(data)
    except Exception:
        import mimetypes
        if fallback_name:
            mt, _ = mimetypes.guess_type(fallback_name)
            return mt or "application/octet-stream"
        return "application/octet-stream"


@dataclass
class ValidationConfig:
    max_bytes_per_file: int = 50 * 1024 * 1024
    max_pages_pdf: int = 500
    max_rows_xlsx: int = 200000
    max_expand_bytes_archive: int = 200 * 1024 * 1024
    max_files_archive: int = 5000
    allowed_mime: Set[str] = None  # supports globs like text/*
    blocked_ext: Set[str] = None

    def __post_init__(self) -> None:
        if self.allowed_mime is None:
            self.allowed_mime = {
                "text/*",
                "application/pdf",
                "application/json",
                "application/zip",
                "application/x-tar",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            }
        if self.blocked_ext is None:
            self.blocked_ext = {".exe", ".dll", ".bin"}


class ValidationError(Exception):
    pass


def _mime_allowed(mime: str, allowed: Iterable[str]) -> bool:
    for pattern in allowed:
        if pattern.endswith("/*"):
            prefix = pattern[:-2]
            if mime.startswith(prefix + "/"):
                return True
        if mime == pattern:
            return True
    return False


def validate_file(key: str, data: bytes, cfg: ValidationConfig) -> str:
    """Validate a single file's raw bytes. Returns detected MIME."""
    # extension block
    _, ext = os.path.splitext(key.lower())
    if ext in (cfg.blocked_ext or set()):
        raise ValidationError(f"Blocked extension: {ext}")

    # size
    if len(data) > cfg.max_bytes_per_file:
        raise ValidationError(f"File too large: {len(data)} > {cfg.max_bytes_per_file}")

    # mime
    mime = _sniff_mime(data, fallback_name=key)
    if not _mime_allowed(mime, cfg.allowed_mime):
        raise ValidationError(f"MIME not allowed: {mime}")

    # archive guard
    if mime in {"application/zip", "application/x-zip-compressed"} or key.lower().endswith(".zip"):
        _validate_zip_bomb(data, cfg)
    if mime in {"application/x-tar"} or key.lower().endswith((".tar", ".tar.gz", ".tgz")):
        _validate_tar_bomb(data, cfg)

    return mime


def _validate_zip_bomb(data: bytes, cfg: ValidationConfig) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            total = 0
            count = 0
            for zi in zf.infolist():
                # basic traversal check
                if os.path.isabs(zi.filename) or ".." in zi.filename.replace("\\", "/").split("/"):
                    raise ValidationError("Unsafe path in ZIP entry")
                count += 1
                if count > cfg.max_files_archive:
                    raise ValidationError("Too many files in archive")
                # expanded size may be missing; use file_size compressed as lower bound
                total += max(zi.file_size, zi.compress_size)
                if total > cfg.max_expand_bytes_archive:
                    raise ValidationError("Archive expanded size too large")
    except zipfile.BadZipFile:
        raise ValidationError("Invalid ZIP archive")


def _validate_tar_bomb(data: bytes, cfg: ValidationConfig) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            total = 0
            count = 0
            for ti in tf.getmembers():
                name = ti.name
                if os.path.isabs(name) or ".." in name.replace("\\", "/").split("/"):
                    raise ValidationError("Unsafe path in TAR entry")
                if ti.isdir():
                    continue
                count += 1
                if count > cfg.max_files_archive:
                    raise ValidationError("Too many files in archive")
                total += max(ti.size, 0)
                if total > cfg.max_expand_bytes_archive:
                    raise ValidationError("Archive expanded size too large")
    except tarfile.TarError:
        raise ValidationError("Invalid TAR archive")


