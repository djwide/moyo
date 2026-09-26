"""Grouping and review-label parsing for document imports."""

from __future__ import annotations

import json
from pathlib import Path

from reports.pipeline.documents import document_to_text
from reports.pipeline.organize import (
    apply_document_graph,
    organize_claims,
    parse_review_labels,
    parse_sections,
    suggest_review_labels,
)


KNOWN = {"C0001", "C0002", "C0003", "C0004"}


def test_confident_section_is_kept_and_singleton_is_dropped():
    text = json.dumps(
        {
            "sections": [
                {"title": "Bank accounts", "claim_ids": ["C0001", "C0004"]},
                {"title": "Lone date", "claim_ids": ["C0002"]},
            ]
        }
    )
    assert parse_sections(text, KNOWN) == [
        {"title": "Bank accounts", "claim_ids": ["C0001", "C0004"]}
    ]


def test_overlap_rejects_the_grouping():
    text = json.dumps(
        {
            "sections": [
                {"title": "One", "claim_ids": ["C0001", "C0002"]},
                {"title": "Two", "claim_ids": ["C0002", "C0003"]},
            ]
        }
    )
    assert parse_sections(text, KNOWN) is None


def test_bad_json_falls_back_to_no_sections():
    assert parse_sections("not json", KNOWN) is None
    assert organize_claims(
        [
            {"claim_id": "C0001", "claim": "Acme opened an office."},
            {"claim_id": "C0002", "claim": "Acme hired a treasurer."},
        ],
        client=type("Boom", (), {"complete": staticmethod(lambda _prompt: "{")})(),
    ) == []


def test_review_labels_skip_useful_and_unknowns():
    text = json.dumps(
        {
            "labels": [
                {"claim_id": "C0001", "label": "investigate"},
                {"claim_id": "C0002", "label": "useful"},
                {"claim_id": "C9999", "label": "known"},
            ]
        }
    )
    assert parse_review_labels(text, KNOWN) == {"C0001": "investigate"}


def test_apply_document_graph_keeps_claims_when_grouping_fails(tmp_path: Path):
    report = {
        "findings_all": [
            {"claim_id": "C0001", "claim": "Acme exists."},
            {"claim_id": "C0002", "claim": "Acme hired a treasurer."},
        ]
    }
    path = tmp_path / "report_data.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    class Boom:
        def complete(self, _prompt: str) -> str:
            raise RuntimeError("model down")

    apply_document_graph(tmp_path, auto_label=True, client=Boom())
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["findings_all"] == report["findings_all"]
    assert saved["sections"] == []


def test_apply_document_graph_writes_sections_and_optional_labels(tmp_path: Path):
    report = {
        "findings_all": [
            {"claim_id": "C0001", "claim": "Acme opened a Bank of China account."},
            {"claim_id": "C0002", "claim": "The account held $100,000."},
            {"claim_id": "C0003", "claim": "Acme sponsored a race."},
        ]
    }
    path = tmp_path / "report_data.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    calls = {"n": 0}

    class Scripted:
        def complete(self, prompt: str) -> str:
            calls["n"] += 1
            if "Suggest a review label" in prompt:
                return json.dumps(
                    {"labels": [{"claim_id": "C0003", "label": "not_relevant"}, {"claim_id": "C0001", "label": "useful"}]}
                )
            return json.dumps(
                {"sections": [{"title": "Bank account", "claim_ids": ["C0001", "C0002"]}]}
            )

    apply_document_graph(tmp_path, auto_label=False, client=Scripted())
    unlabeled = json.loads(path.read_text(encoding="utf-8"))
    assert unlabeled["sections"] == [
        {"title": "Bank account", "claim_ids": ["C0001", "C0002"]}
    ]
    assert "customerLabels" not in unlabeled["findings_all"][0]

    apply_document_graph(tmp_path, auto_label=True, client=Scripted())
    labeled = json.loads(path.read_text(encoding="utf-8"))
    by_id = {row["claim_id"]: row for row in labeled["findings_all"]}
    assert by_id["C0003"]["customerLabels"] == ["not_relevant"]
    assert "customerLabels" not in by_id["C0001"]
    assert suggest_review_labels(
        [{"claim_id": "C0001", "claim": "A fact."}],
        client=type("UsefulOnly", (), {"complete": staticmethod(lambda _prompt: json.dumps({"labels": [{"claim_id": "C0001", "label": "useful"}]}))})(),
    ) == {}


def test_document_to_text_rejects_empty_and_reads_notes():
    assert document_to_text(b"A plain note.", ".txt") == "A plain note."
    try:
        document_to_text(b"   \n", ".md")
    except ValueError as exc:
        assert "No text" in str(exc)
    else:
        raise AssertionError("blank notes must fail")


def test_pdf_and_docx_use_extractors(monkeypatch):
    import reports.pipeline.documents as documents

    monkeypatch.setattr(documents, "_pdf_text", lambda raw: "Page one" if raw == b"pdf" else "")
    monkeypatch.setattr(documents, "_docx_text", lambda raw: "Paragraph" if raw == b"docx" else "")
    assert document_to_text(b"pdf", ".pdf") == "Page one"
    assert document_to_text(b"docx", ".docx") == "Paragraph"
    try:
        document_to_text(b"pdf-empty", ".pdf")
    except ValueError as exc:
        assert "No text" in str(exc)
    else:
        raise AssertionError("empty pdf text must fail")
