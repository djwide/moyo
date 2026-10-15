"""3–5 word claim summaries for map pods."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reports.pipeline.summarize import (
    attach_claim_summaries,
    claim_node_needs_summary,
    clip_claim_summary,
    normalize_summary,
    parse_summaries,
    summarize_backfill_nodes,
)


def test_normalize_summary_keeps_three_to_five_words():
    assert normalize_summary("  Fingerstick blood tests. ") == "Fingerstick blood tests"
    assert normalize_summary("one two") is None
    assert normalize_summary("one two three four five six") is None
    assert normalize_summary("Wait. More words here") is None


def test_normalize_summary_keeps_a_short_spaceless_phrase():
    assert normalize_summary("指先採血検査") == "指先採血検査"


def test_parse_summaries_drops_unknown_and_invalid_rows():
    text = json.dumps(
        {
            "summaries": [
                {"claim_id": "C0001", "summary": "Fingerstick blood tests"},
                {"claim_id": "C0002", "summary": "too short"},
                {"claim_id": "NOPE", "summary": "Invented claim label"},
            ]
        }
    )
    assert parse_summaries(text, {"C0001", "C0002"}) == {
        "C0001": "Fingerstick blood tests"
    }


def test_attach_retries_invalid_summaries_once():
    calls = {"n": 0}

    class Client:
        def complete(self, prompt: str, **_kwargs: object) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                return json.dumps(
                    {"summaries": [{"claim_id": "C0001", "summary": "not enough"}]}
                )
            return json.dumps(
                {"summaries": [{"claim_id": "C0001", "summary": "Fingerstick blood tests"}]}
            )

    claims = [{"claim_id": "C0001", "claim": "The device ran hundreds of tests from a fingerstick."}]
    assert attach_claim_summaries(claims, client=Client()) == 1
    assert claims[0]["summary"] == "Fingerstick blood tests"
    assert calls["n"] == 2


def test_attach_leaves_summary_unset_after_a_failed_retry():
    class Client:
        def complete(self, _prompt: str, **_kwargs: object) -> str:
            return json.dumps({"summaries": [{"claim_id": "C0001", "summary": "no"}]})

    claims = [{"claim_id": "C0001", "claim": "A long claim that needs a label."}]
    assert attach_claim_summaries(claims, client=Client()) == 0
    assert "summary" not in claims[0]


def test_attach_does_not_overwrite_an_existing_summary():
    class Client:
        def complete(self, _prompt: str, **_kwargs: object) -> str:
            raise AssertionError("should not call the model")

    claims = [{"claim_id": "C0001", "claim": "Already labeled.", "summary": "Kept custom label"}]
    assert attach_claim_summaries(claims, client=Client()) == 0
    assert claims[0]["summary"] == "Kept custom label"


def test_dry_run_clips_without_a_model():
    claims = [
        {
            "claim_id": "C0001",
            "claim": "Theranos said its devices could run hundreds of blood tests from a fingerstick.",
        }
    ]
    assert attach_claim_summaries(claims, dry_run=True) == 1
    assert claims[0]["summary"] == clip_claim_summary(claims[0]["claim"])
    assert len(claims[0]["summary"].split()) == 5


def test_backfill_skips_sections_and_existing_summaries():
    assert claim_node_needs_summary("section_abc", {"claim": "Banking group"}) is False
    assert claim_node_needs_summary("n1", {"claim": "A fact", "summary": "Already set here"}) is False
    assert claim_node_needs_summary("n1", {"claim": "A fact", "finding": {"section": True}}) is False
    assert claim_node_needs_summary("n1", {"claim": "A sourced fact"}) is True

    updates = summarize_backfill_nodes(
        [
            {"project_id": "map1", "id": "n1", "claim": "Devices ran fingerstick blood tests in stores."},
            {"project_id": "map1", "id": "section_abc", "claim": "Banking"},
            {"project_id": "map1", "id": "n2", "claim": "Already done.", "summary": "Kept short label"},
        ],
        dry_run=True,
    )
    assert updates == [
        {
            "project_id": "map1",
            "id": "n1",
            "summary": "Devices ran fingerstick blood tests",
        }
    ]


def test_backfill_write_sets_summary_only():
    import sys

    webpage = Path(__file__).resolve().parents[2] / "moyomapwebpage"
    if not (webpage / "backfill_claim_summaries.py").is_file():
        pytest.skip("moyomapwebpage is not checked out next to moyo")
    sys.path.insert(0, str(webpage))
    from backfill_claim_summaries import write_summaries

    class FakeBatch:
        def __init__(self, db: "FakeDb") -> None:
            self.db = db
            self.ops: list[tuple[str, dict[str, str]]] = []

        def update(self, ref: "FakeRef", data: dict[str, str]) -> None:
            self.ops.append((ref.path, data))

        def commit(self) -> None:
            self.db.commits.extend(self.ops)
            self.ops = []

    class FakeRef:
        def __init__(self, path: str) -> None:
            self.path = path

        def collection(self, name: str) -> "FakeCollection":
            return FakeCollection(f"{self.path}/{name}")

    class FakeCollection:
        def __init__(self, path: str) -> None:
            self.path = path

        def document(self, doc_id: str) -> FakeRef:
            return FakeRef(f"{self.path}/{doc_id}")

    class FakeDb:
        def __init__(self) -> None:
            self.commits: list[tuple[str, dict[str, str]]] = []

        def batch(self) -> FakeBatch:
            return FakeBatch(self)

        def collection(self, name: str) -> FakeCollection:
            return FakeCollection(name)

    db = FakeDb()
    written = write_summaries(
        db,
        [{"project_id": "map1", "id": "n1", "summary": "Fingerstick blood tests"}],
    )
    assert written == 1
    assert db.commits == [
        ("moyomap_projects/map1/nodes/n1", {"summary": "Fingerstick blood tests"})
    ]


def test_compact_finding_includes_summary():
    import cloud_worker as cw

    compact = cw.compact_finding(
        {
            "claim_id": "C0001",
            "claim": "Full claim text.",
            "citations": ["https://example.com"],
            "summary": "Fingerstick blood tests",
        }
    )
    assert compact["summary"] == "Fingerstick blood tests"
    assert compact["claim"] == "Full claim text."
    assert compact["citations"] == ["https://example.com"]
