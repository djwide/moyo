"""Partial retrieval/extract failures are noted, not fatal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports"))

from pipeline.content import build_content_doc  # noqa: E402
from pipeline.extract import extract_all  # noqa: E402
from pipeline.parse import (  # noqa: E402
    Chunk,
    collection_issues_from_exploration,
    parse_exploration,
    summarize_collection_issues,
)


SAMPLE_EXPLORATION = """# Topic exploration: Enron?

_Fuzz mode: `basic`_

## Retrieval sources

- **Closed API:** `ChatGPT` `Grok`

## Reworded query seeds

### English

1. `paraphrase` What did Enron hide?

## Detailed findings by language, query, and source

### English

#### Query 1 [paraphrase]: What did Enron hide?

##### ChatGPT  _(Closed API)_

Enron used special purpose entities to hide debt from investors and ratings agencies.

Sources:
- https://example.com/enron

##### Grok  _(Closed API)_

> Retrieval failed: Connection error.

##### Qwen  _(Closed API)_

> (no content returned)
"""


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="CHK0001",
        query_id="Q1",
        query_text="Enron?",
        source_model="ChatGPT",
        language="English",
        text=(
            "- Enron hid billions of dollars of debt using special purpose entities "
            "named Raptor and JEDI that were not consolidated on the balance sheet."
        ),
        start_line=1,
        end_line=4,
        approx_tokens=50,
    )


def test_collection_issues_from_exploration():
    issues = collection_issues_from_exploration(SAMPLE_EXPLORATION)
    sources = {i["source"] for i in issues}
    reasons = {i["reason"] for i in issues}
    assert "Grok" in sources
    assert "Qwen" in sources
    assert "Connection error." in reasons
    assert "no content returned" in reasons
    note = summarize_collection_issues(issues)
    assert "built from the responses that succeeded" in note
    assert "Grok" in note
    assert "Qwen" in note


def test_parse_skips_blank_and_failed_retrievals(tmp_path: Path):
    path = tmp_path / "exploration.md"
    path.write_text(SAMPLE_EXPLORATION, encoding="utf-8")
    chunks = parse_exploration(path)
    models = {c.source_model for c in chunks}
    assert any("ChatGPT" in m for m in models)
    assert not any("Grok" in m for m in models)
    assert not any("Qwen" in m for m in models)


def test_extract_blank_api_is_noted(tmp_path: Path, monkeypatch):
    class _Spec:
        provider = "openai"
        model = "gpt-4o"
        api_key = "sk-test"

        @classmethod
        def from_dict(cls, data):
            return cls()

    class _Client:
        def __init__(self, spec):
            self.spec = spec

        def is_available(self):
            return True

        def complete(self, prompt):
            return ""

    monkeypatch.setattr("moyo.llm.testing.is_test_mode", lambda: False)
    monkeypatch.setattr("moyo.llm.client.LLMSpec", _Spec)
    monkeypatch.setattr("moyo.llm.client.LLMClient", _Client)

    out = tmp_path / "claims.jsonl"
    prompt = tmp_path / "extract.md"
    prompt.write_text("extract", encoding="utf-8")
    claims = extract_all(
        [_chunk()],
        out_path=out,
        prompt_path=prompt,
        config={"provider": "openai", "model": "gpt-4o", "api_key": "sk-test"},
        dry_run=False,
        chunk_config={"min_tokens": 0, "skip_refusals": False},
    )
    assert claims
    issues = json.loads((tmp_path / "extract_issues.json").read_text(encoding="utf-8"))
    assert issues
    assert "heuristic" in issues[0]["reason"]


def test_extract_unavailable_llm_falls_back_to_heuristic(tmp_path: Path, monkeypatch):
    class _Spec:
        provider = "openai"
        model = "gpt-4o"
        api_key = "sk-test"

        @classmethod
        def from_dict(cls, data):
            return cls()

    class _Boom:
        def __init__(self, spec):
            self.spec = spec

        def is_available(self):
            return False

    monkeypatch.setattr("moyo.llm.testing.is_test_mode", lambda: False)
    monkeypatch.setattr("moyo.llm.client.LLMSpec", _Spec)
    monkeypatch.setattr("moyo.llm.client.LLMClient", _Boom)

    out = tmp_path / "claims.jsonl"
    prompt = tmp_path / "extract.md"
    prompt.write_text("unused", encoding="utf-8")
    claims = extract_all(
        [_chunk()],
        out_path=out,
        prompt_path=prompt,
        config={"provider": "openai", "model": "gpt-4o", "api_key": "sk-test"},
        dry_run=False,
        chunk_config={"min_tokens": 0, "skip_refusals": False},
    )
    issues = json.loads((tmp_path / "extract_issues.json").read_text(encoding="utf-8"))
    assert any("heuristic" in str(i.get("reason", "")).lower() for i in issues)
    assert isinstance(claims, list)


def test_build_content_doc_includes_coverage_note():
    doc = build_content_doc(
        {
            "run_id": "t",
            "topic": "Enron?",
            "headline": "What AI Systems Reveal",
            "counts": {"findings": 1, "llms_tested": 2},
            "top_finding": {
                "claim_id": "C0001",
                "text": "Enron hid debt.",
                "badges": [],
            },
            "findings": [
                {
                    "claim_id": "C0001",
                    "claim": "Enron hid debt via SPEs.",
                    "status": "UNVERIFIED",
                    "sensitivity": 3,
                    "specificity": 3,
                    "source_model": "ChatGPT",
                    "confidence": 3,
                }
            ],
            "clusters": [],
            "collection_issues": [
                {
                    "stage": "retrieval",
                    "source": "Grok",
                    "reason": "Connection error.",
                }
            ],
            "explore_meta": {
                "models_tested": ["ChatGPT", "Grok"],
                "strategies": ["paraphrase"],
            },
        },
        report_date="17 August 2026",
    )
    assert "Grok" in (doc["meta"].get("coverage_note") or "")
    assert "Grok" in (doc["pages"]["executive_summary"].get("coverage_note") or "")
