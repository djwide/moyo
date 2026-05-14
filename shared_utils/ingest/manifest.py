"""Manifest schema and writer ensuring idempotent ingestion records."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..ids import generate_content_hash


@dataclass
class ChunkRecord:
    id: str
    rel_path: str
    len: int


@dataclass
class ManifestRecord:
    ingest_id: str
    src_path: str
    stored_at: str
    mime: str
    size_bytes: int
    bytes_sha256: str
    text_sha256: str
    loader: str
    normalized_at: str
    language: str
    policy_tags: List[str]
    notes: str
    chunks: List[ChunkRecord]


class ManifestWriter:
    def __init__(self, manifest_key: str, storage) -> None:
        self.manifest_key = manifest_key
        self.storage = storage

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_existing_hashes(self) -> set[str]:
        if not self.storage.exists(self.manifest_key):
            return set()
        hashes: set[str] = set()
        data = self.storage.read_bytes(self.manifest_key).decode("utf-8", "ignore")
        for line in data.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                h = obj.get("bytes_sha256")
                if h:
                    hashes.add(h)
            except Exception:
                continue
        return hashes

    def append_if_new(self, rec: ManifestRecord) -> bool:
        """Append record if bytes_sha256 not present. Returns True if written."""
        existing = self._load_existing_hashes()
        if rec.bytes_sha256 in existing:
            return False
        line = json.dumps(asdict(rec), ensure_ascii=False)
        # append
        prev = b""
        if self.storage.exists(self.manifest_key):
            prev = self.storage.read_bytes(self.manifest_key)
            if prev and not prev.endswith(b"\n"):
                prev += b"\n"
        self.storage.write_bytes(self.manifest_key, prev + line.encode("utf-8") + b"\n")
        return True


