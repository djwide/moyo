"""Records for the local sensitive-phrases corpus."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

PhraseStatus = Literal["pending", "approved", "rejected"]
PhraseSource = Literal["document", "manual"]

LABELS = (
    "credential",
    "identifier",
    "financial",
    "project",
    "personnel",
    "operational",
    "other",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def phrase_id(text: str) -> str:
    key = re.sub(r"\s+", " ", (text or "").strip()).lower().encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()
    return f"ph_{digest[:12]}"


@dataclass
class PhraseRecord:
    """One candidate or approved phrase."""

    id: str
    text: str
    label: str = "other"
    status: PhraseStatus = "pending"
    source: PhraseSource = "document"
    source_path: str | None = None
    reason: str = ""
    score: float = 0.0
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhraseRecord":
        return cls(
            id=str(data.get("id") or ""),
            text=str(data.get("text") or "").strip(),
            label=str(data.get("label") or "other").strip() or "other",
            status=data.get("status") or "pending",  # type: ignore[arg-type]
            source=data.get("source") or "document",  # type: ignore[arg-type]
            source_path=(str(data["source_path"]) if data.get("source_path") else None),
            reason=str(data.get("reason") or ""),
            score=float(data.get("score") or 0.0),
            created_at=str(data.get("created_at") or utc_now()),
        )
