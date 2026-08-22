"""Local JSONL store for pending + approved sensitive phrases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from moyo.privateside.phrases.schema import PhraseRecord, phrase_id, utc_now

# Legacy path — do not use. Phrases live in projects/<name>/phrases/.
DEFAULT_CORPUS_DIR = Path("data/private/phrases")


class PhraseStore:
    """``pending.jsonl`` for review, ``corpus.jsonl`` + ``corpus.txt`` approved."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.pending_path = self.root / "pending.jsonl"
        self.corpus_path = self.root / "corpus.jsonl"
        self.txt_path = self.root / "corpus.txt"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def load_pending(self) -> list[PhraseRecord]:
        return _read_jsonl(self.pending_path)

    def load_approved(self) -> list[PhraseRecord]:
        return [p for p in _read_jsonl(self.corpus_path) if p.status == "approved"]

    def index_items(self) -> list[dict]:
        """Approved phrases as FAISS corpus rows."""
        items = []
        for i, rec in enumerate(self.load_approved()):
            items.append(
                {
                    "id": rec.id,
                    "text": rec.text,
                    "source": rec.source_path or rec.source,
                    "chunk_id": i,
                    "label": rec.label,
                }
            )
        return items

    def known_keys(self) -> set[str]:
        keys: set[str] = set()
        for rec in _read_jsonl(self.pending_path) + _read_jsonl(self.corpus_path):
            if rec.text:
                keys.add(_key(rec.text))
            if rec.id:
                keys.add(rec.id)
        return keys

    def enqueue(self, records: Iterable[PhraseRecord]) -> list[PhraseRecord]:
        """Append unseen pending records. Returns those actually added."""
        self.ensure()
        known = self.known_keys()
        added: list[PhraseRecord] = []
        with self.pending_path.open("a", encoding="utf-8") as fh:
            for rec in records:
                if not rec.text:
                    continue
                key = _key(rec.text)
                if key in known or rec.id in known:
                    continue
                rec.status = "pending"
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                known.add(key)
                known.add(rec.id)
                added.append(rec)
        return added

    def add_manual(self, text: str, label: str = "other") -> PhraseRecord | None:
        from moyo.privateside.phrases.schema import PhraseRecord, phrase_id

        cleaned = " ".join((text or "").split())
        if not cleaned:
            return None
        rec = PhraseRecord(
            id=phrase_id(cleaned),
            text=cleaned,
            label=(label or "other").strip() or "other",
            status="approved",
            source="manual",
            reason="manual",
            score=1.0,
        )
        if _key(rec.text) in self.known_keys() or rec.id in self.known_keys():
            return None
        self._append_approved([rec])
        return rec

    def add_manual_lines(self, lines: Iterable[str], default_label: str = "other") -> list[PhraseRecord]:
        added: list[PhraseRecord] = []
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            phrase, label = _split_phrase_label(line, default_label)
            rec = self.add_manual(phrase, label)
            if rec:
                added.append(rec)
        return added

    def decide(self, record_id: str, *, approve: bool, label: str | None = None) -> PhraseRecord | None:
        pending = self.load_pending()
        match = next((p for p in pending if p.id == record_id), None)
        if match is None:
            return None
        remaining = [p for p in pending if p.id != record_id]
        _write_jsonl(self.pending_path, remaining)
        if approve:
            match.status = "approved"
            if label and label.strip():
                match.label = label.strip()
            match.created_at = utc_now()
            self._append_approved([match])
        else:
            match.status = "rejected"
            self.ensure()
            rejected = self.root / "rejected.jsonl"
            with rejected.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(match.to_dict(), ensure_ascii=False) + "\n")
        return match

    def _append_approved(self, records: list[PhraseRecord]) -> None:
        self.ensure()
        with self.corpus_path.open("a", encoding="utf-8") as fh:
            for rec in records:
                rec.status = "approved"
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        existing = {
            line.strip()
            for line in self.txt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        } if self.txt_path.is_file() else set()
        with self.txt_path.open("a", encoding="utf-8") as fh:
            for rec in records:
                if rec.text not in existing:
                    fh.write(rec.text + "\n")
                    existing.add(rec.text)


def _key(text: str) -> str:
    return " ".join(text.lower().split())


def _split_phrase_label(line: str, default_label: str) -> tuple[str, str]:
    if "|" in line:
        left, right = line.split("|", 1)
        return left.strip(), (right.strip() or default_label)
    return line.strip(), default_label


def _read_jsonl(path: Path) -> list[PhraseRecord]:
    if not path.is_file():
        return []
    out: list[PhraseRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("text"):
            out.append(PhraseRecord.from_dict(data))
    return out


def load_corpus_for_index(path: Path | str) -> list[dict]:
    """Load approved phrases or a legacy GUI corpus file for FAISS."""
    src = Path(path)
    if not src.is_file():
        return []
    if src.suffix.lower() == ".jsonl":
        return [
            {
                "id": rec.id,
                "text": rec.text,
                "source": rec.source_path or rec.source,
                "chunk_id": i,
                "label": rec.label,
            }
            for i, rec in enumerate(_read_jsonl(src))
            if rec.status != "rejected" and rec.text
        ]
    text = src.read_text(encoding="utf-8")
    if "ID: " in text and "Text: " in text:
        items = []
        for section in text.split("-" * 50):
            if not section.strip():
                continue
            item: dict = {}
            for line in section.strip().split("\n"):
                if line.startswith("ID: "):
                    item["id"] = line[4:].strip()
                elif line.startswith("Source: "):
                    item["source"] = line[8:].strip()
                elif line.startswith("Text: "):
                    item["text"] = line[6:].strip()
            if item.get("text", "").strip():
                items.append(item)
        return items
    return [
        {"id": f"line_{i}", "text": line.strip(), "source": src.name, "chunk_id": i}
        for i, line in enumerate(text.splitlines())
        if line.strip() and not line.startswith("#")
    ]


def _write_jsonl(path: Path, records: list[PhraseRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
