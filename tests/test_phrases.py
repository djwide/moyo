from __future__ import annotations

import json
from pathlib import Path

from moyo.privateside.phrases.filter import extract_sensitive_phrases, parse_phrase_payload
from moyo.privateside.phrases.ingest import ingest_document, ingest_text
from moyo.privateside.phrases.store import PhraseStore


def test_parse_kimi_json_payload():
    raw = """```json
    {"phrases": [
      {"text": "Project FOXTROT launch is 2026-09-01", "label": "project", "reason": "codename"},
      {"text": "", "label": "other"},
      {"text": "alice@acme.internal", "label": "mystery"}
    ]}
    ```"""
    rows = parse_phrase_payload(raw)
    assert len(rows) == 2
    assert rows[0]["label"] == "project"
    assert rows[1]["label"] == "other"


def _stub_complete(prompt, system=None):
    return json.dumps(
        {
            "phrases": [
                {
                    "text": "Project FOXTROT launch is 2026-09-01 in Building 4",
                    "label": "project",
                    "reason": "codename and site",
                },
                {
                    "text": "alice@acme.internal",
                    "label": "identifier",
                    "reason": "internal email",
                },
            ]
        }
    )


def test_ingest_uses_kimi_stub_not_heuristics(tmp_path: Path):
    doc = tmp_path / "memo.txt"
    doc.write_text(
        "Confidential\n\n"
        "This document is for internal use only.\n\n"
        "Project FOXTROT launch is 2026-09-01 in Building 4.\n\n"
        "Contact alice@acme.internal for badge access.\n",
        encoding="utf-8",
    )
    store = PhraseStore(tmp_path / "phrases")
    result = ingest_document(doc, store, complete=_stub_complete)
    assert result["queued"] == 2
    texts = [p.text.lower() for p in store.load_pending()]
    assert any("foxtrot" in t for t in texts)
    assert any("alice@acme.internal" in t for t in texts)
    assert not any("this document is for internal use only" in t for t in texts)


def test_extract_appends_labelled_direction():
    seen = {}

    def capture(prompt, system=None):
        seen["prompt"] = prompt
        return json.dumps({"phrases": []})

    extract_sensitive_phrases(
        "Project FOXTROT is secret.",
        direction="Focus on personnel names only.",
        complete=capture,
    )
    assert "Project FOXTROT is secret." in seen["prompt"]
    assert "\ndirection:\nFocus on personnel names only." in seen["prompt"]
    assert seen["prompt"].index("Project FOXTROT is secret.") < seen["prompt"].index("direction:")


def test_extract_omits_empty_direction():
    seen = {}

    def capture(prompt, system=None):
        seen["prompt"] = prompt
        return json.dumps({"phrases": []})

    extract_sensitive_phrases("Project FOXTROT is secret.", direction="  ", complete=capture)
    assert "direction:" not in seen["prompt"]


def test_ingest_text_direct(tmp_path: Path):
    store = PhraseStore(tmp_path / "phrases")
    result = ingest_text(
        "Project FOXTROT is secret.",
        store,
        source_path="paste",
        direction="Keep project codenames.",
        complete=_stub_complete,
    )
    assert result["queued"] == 2


def test_manual_add_and_index_items(tmp_path: Path):
    store = PhraseStore(tmp_path / "phrases")
    rec = store.add_manual("codename NIGHTSHADE", "project")
    assert rec is not None
    items = store.index_items()
    assert items[0]["text"] == "codename NIGHTSHADE"
    assert items[0]["label"] == "project"
    txt = (tmp_path / "phrases" / "corpus.txt").read_text(encoding="utf-8")
    assert "codename NIGHTSHADE" in txt


def test_review_approve_moves_off_pending(tmp_path: Path):
    store = PhraseStore(tmp_path / "phrases")
    ingest_text("secret", store, complete=_stub_complete)
    pending = store.load_pending()
    assert pending
    for rec in pending:
        store.decide(rec.id, approve=True, label="project")
    assert store.load_pending() == []
    assert store.index_items()[0]["label"] == "project"


def test_chunk_for_phrases_uses_sentences():
    from moyo.privateside.phrases.ingest import chunk_for_phrases

    chunks = chunk_for_phrases(
        "Alpha is one fact. Beta is another fact with a date 2026-01-01."
    )
    assert any("Alpha is one fact" in c for c in chunks)
    assert any("Beta is another" in c for c in chunks)
