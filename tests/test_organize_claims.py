"""Grouping and review-label parsing for document imports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reports.pipeline.documents import document_to_text
from reports.pipeline.organize import (
    apply_document_graph,
    filter_claims_for_theme,
    layout_document_claims,
    organize_claims,
    parse_attach_parents,
    parse_claim_parents,
    parse_document_layout,
    parse_review_labels,
    parse_sections,
    parse_theme_filter,
    suggest_review_labels,
    tree_claims,
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
        client=type("Boom", (), {"complete": staticmethod(lambda _prompt, **_kw: "{")})(),
    ) == []


def test_claim_parents_accept_confident_links_and_reject_cycles():
    text = json.dumps(
        {
            "parents": [
                {"claim_id": "C0002", "parent_claim_id": "C0001"},
                {"claim_id": "C0003", "parent_claim_id": "C0002"},
            ]
        }
    )
    assert parse_claim_parents(text, KNOWN) == {"C0002": "C0001", "C0003": "C0002"}
    assert (
        parse_claim_parents(
            json.dumps(
                {
                    "parents": [
                        {"claim_id": "C0001", "parent_claim_id": "C0002"},
                        {"claim_id": "C0002", "parent_claim_id": "C0001"},
                    ]
                }
            ),
            KNOWN,
        )
        is None
    )
    assert (
        parse_claim_parents(
            json.dumps({"parents": [{"claim_id": "C0001", "parent_claim_id": "C0001"}]}),
            KNOWN,
        )
        is None
    )


def test_parse_attach_parents_keeps_new_children_only():
    text = json.dumps(
        {
            "parents": [
                {"claim_id": "C0001", "parent_claim_id": "claim_office"},
                {"claim_id": "claim_office", "parent_claim_id": "C0001"},
            ]
        }
    )
    assert parse_attach_parents(text, {"C0001"}, {"C0001", "claim_office"}) == {
        "C0001": "claim_office"
    }


def test_tree_claims_skips_bad_batches():
    assert (
        tree_claims(
            [
                {"claim_id": "C0001", "claim": "Acme opened an office."},
                {"claim_id": "C0002", "claim": "The office is in Cayman."},
            ],
            client=type("Boom", (), {"complete": staticmethod(lambda _prompt, **_kw: "{")})(),
        )
        == {}
    )


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


def test_apply_document_graph_requires_utility_when_requested(tmp_path: Path, monkeypatch):
    report = {
        "findings_all": [
            {"claim_id": "C0001", "claim": "Acme exists."},
            {"claim_id": "C0002", "claim": "Acme hired a treasurer."},
        ]
    }
    path = tmp_path / "report_data.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        "reports.pipeline.organize.utility_organize_client",
        lambda: None,
    )
    with pytest.raises(RuntimeError, match="Utility LLM"):
        apply_document_graph(tmp_path, auto_label=False, require_utility=True)


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
        def complete(self, _prompt: str, **_kw: object) -> str:
            raise RuntimeError("model down")

    apply_document_graph(tmp_path, auto_label=True, client=Boom())
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["findings_all"] == report["findings_all"]
    assert "sections" not in saved
    assert "claim_parents" not in saved
    layout = json.loads((tmp_path / "tree_layout.json").read_text(encoding="utf-8"))
    assert "edges" not in layout
    assert "sections" not in layout
    assert "claim_parents" not in layout
    by_id = {node["id"]: node for node in layout["nodes"]}
    assert by_id["C0001"]["parent"] == "root"
    assert by_id["C0001"]["depth"] == 1
    assert by_id["C0002"]["parent"] == "root"
    assert by_id["C0002"]["depth"] == 1
    assert layout["review_labels"] == {}


def test_apply_document_graph_writes_sections_tree_and_optional_labels(tmp_path: Path):
    report = {
        "findings_all": [
            {"claim_id": "C0001", "claim": "Acme opened a Bank of China account."},
            {"claim_id": "C0002", "claim": "The account held $100,000."},
            {"claim_id": "C0003", "claim": "Acme sponsored a race."},
        ]
    }
    path = tmp_path / "report_data.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    class Scripted:
        def complete(self, prompt: str, **_kw: object) -> str:
            if "Suggest a review label" in prompt:
                return json.dumps(
                    {
                        "labels": [
                            {"claim_id": "C0003", "label": "not_relevant"},
                            {"claim_id": "C0001", "label": "useful"},
                        ]
                    }
                )
            if "Nest extracted claims" in prompt or "parent_claim_id" in prompt:
                return json.dumps(
                    {"parents": [{"claim_id": "C0002", "parent_claim_id": "C0001"}]}
                )
            return json.dumps(
                {"sections": [{"title": "Bank account", "claim_ids": ["C0001", "C0002"]}]}
            )

    apply_document_graph(tmp_path, auto_label=False, client=Scripted())
    unlabeled = json.loads(path.read_text(encoding="utf-8"))
    assert unlabeled["findings_all"] == report["findings_all"]
    assert "sections" not in unlabeled
    layout = json.loads((tmp_path / "tree_layout.json").read_text(encoding="utf-8"))
    assert "edges" not in layout
    by_id = {node["id"]: node for node in layout["nodes"]}
    assert by_id["section:0"] == {
        "id": "section:0",
        "kind": "section",
        "title": "Bank account",
        "parent": "root",
        "depth": 1,
    }
    assert by_id["C0001"]["parent"] == "section:0"
    assert by_id["C0001"]["depth"] == 2
    assert by_id["C0002"]["parent"] == "C0001"
    assert by_id["C0002"]["depth"] == 3
    assert by_id["C0003"]["parent"] == "root"
    assert by_id["C0003"]["depth"] == 1
    assert layout["review_labels"] == {}

    apply_document_graph(tmp_path, auto_label=True, client=Scripted())
    still_unlabeled = json.loads(path.read_text(encoding="utf-8"))
    assert still_unlabeled["findings_all"] == report["findings_all"]
    labeled_layout = json.loads((tmp_path / "tree_layout.json").read_text(encoding="utf-8"))
    assert labeled_layout["review_labels"] == {"C0003": "not_relevant"}
    assert (
        suggest_review_labels(
            [{"claim_id": "C0001", "claim": "A fact."}],
            client=type(
                "UsefulOnly",
                (),
                {
                    "complete": staticmethod(
                        lambda _prompt, **_kw: json.dumps(
                            {"labels": [{"claim_id": "C0001", "label": "useful"}]}
                        )
                    )
                },
            )(),
        )
        == {}
    )


def test_apply_document_graph_stub_only_skips_utility(tmp_path: Path):
    report = {
        "topic": "Acme",
        "findings_all": [
            {"claim_id": "C0001", "claim": "Acme exists."},
            {"claim_id": "C0002", "claim": "Acme hired a treasurer."},
        ],
    }
    (tmp_path / "report_data.json").write_text(json.dumps(report), encoding="utf-8")

    class Boom:
        def complete(self, _prompt: str, **_kw: object) -> str:
            raise AssertionError("investigate stub must not call the utility LLM")

    apply_document_graph(tmp_path, auto_label=False, client=Boom(), stub_only=True)
    layout = json.loads((tmp_path / "tree_layout.json").read_text(encoding="utf-8"))
    by_id = {node["id"]: node for node in layout["nodes"]}
    assert by_id["C0001"]["parent"] == "root"
    assert by_id["C0002"]["parent"] == "root"
    assert layout["review_labels"] == {}


def test_apply_document_graph_attach_mode_links_to_existing_nodes(tmp_path: Path):
    from reports.pipeline.organize import anchor_nodes_from_prior_claims, place_find_more_claims

    report = {
        "topic": "Acme",
        "findings_all": [
            {"claim_id": "C0001", "claim": "The Cayman office opened in March."},
            {"claim_id": "C0002", "claim": "Acme also hired a treasurer."},
            {"claim_id": "C0003", "claim": "A related payroll vendor was paid."},
        ],
    }
    (tmp_path / "report_data.json").write_text(json.dumps(report), encoding="utf-8")
    existing = [
        {
            "nodeId": "section_bank",
            "claim": "Banking",
            "section": True,
            "depth": 1,
            "parentId": "topic_1",
        },
        {
            "nodeId": "claim_office",
            "claim": "Acme opened an office in Cayman.",
            "section": False,
            "depth": 2,
            "parentId": "section_bank",
        },
    ]

    class Scripted:
        def complete(self, prompt: str, **_kw: object) -> str:
            if "Existing map nodes" in prompt or "newly found" in prompt:
                return json.dumps(
                    {"parents": [{"claim_id": "C0001", "parent_claim_id": "claim_office"}]}
                )
            return json.dumps(
                {"sections": [{"title": "Treasury", "claim_ids": ["C0002", "C0003"]}]}
            )

    apply_document_graph(
        tmp_path,
        auto_label=False,
        client=Scripted(),
        attach=True,
        existing_nodes=anchor_nodes_from_prior_claims(existing),
    )
    layout = json.loads((tmp_path / "tree_layout.json").read_text(encoding="utf-8"))
    by_id = {node["id"]: node for node in layout["nodes"]}
    assert "claim_office" not in by_id
    assert "section_bank" not in by_id
    assert by_id["C0001"]["parent"] == "claim_office"
    assert by_id["C0001"]["depth"] == 3
    assert by_id["C0002"]["parent"] == "section:0"
    assert by_id["C0003"]["parent"] == "section:0"
    assert by_id["section:0"]["title"] == "Treasury"
    assert (
        place_find_more_claims(
            [{"claim_id": "C0001", "claim": "New fact."}],
            [{"claim_id": "claim_office", "claim": "Old fact.", "section": False}],
            client=type("Boom", (), {"complete": staticmethod(lambda _prompt, **_kw: "{")})(),
        )
        == {}
    )


def test_parse_theme_filter_keeps_omissions_and_rejects_conflicts():
    assert parse_theme_filter(
        json.dumps({"keep": ["C0001"], "drop": ["C0002"]}),
        KNOWN,
    ) == {"C0001", "C0003", "C0004"}
    assert parse_theme_filter(
        json.dumps({"keep": ["C0001"], "drop": ["C0001"]}),
        KNOWN,
    ) is None


def test_parse_document_layout_is_one_step_inside_a_section():
    text = json.dumps(
        {
            "sections": [
                {"title": "Bank account", "claim_ids": ["C0001", "C0002", "C0003"]},
            ],
            "parents": [
                {"claim_id": "C0002", "parent_claim_id": "C0001"},
                {"claim_id": "C0003", "parent_claim_id": "C0002"},
            ],
        }
    )
    sections, parents = parse_document_layout(text, KNOWN) or ([], {})
    assert sections == [{"title": "Bank account", "claim_ids": ["C0001", "C0002", "C0003"]}]
    assert parents == {"C0002": "C0001"}


def test_document_import_drops_off_theme_claims_before_layout(tmp_path: Path):
    report = {
        "topic": "Acme",
        "findings_all": [
            {"claim_id": "C0001", "claim": "Acme opened a bank account."},
            {"claim_id": "C0002", "claim": "The account held $100,000."},
            {"claim_id": "C0003", "claim": "A bakery sold bread downtown."},
        ],
    }
    path = tmp_path / "report_data.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    class Scripted:
        def complete(self, prompt: str, **_kw: object) -> str:
            if "Keep claims that bear on one theme" in prompt:
                return json.dumps({"keep": ["C0001", "C0002"], "drop": ["C0003"]})
            if "Lay out claims" in prompt:
                return json.dumps(
                    {
                        "sections": [
                            {"title": "Bank account", "claim_ids": ["C0001", "C0002"]}
                        ],
                        "parents": [{"claim_id": "C0002", "parent_claim_id": "C0001"}],
                    }
                )
            raise AssertionError(prompt[:80])

    apply_document_graph(
        tmp_path,
        auto_label=False,
        client=Scripted(),
        document_import=True,
        theme="Acme banking",
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["findings_all"] == report["findings_all"]
    layout = json.loads((tmp_path / "tree_layout.json").read_text(encoding="utf-8"))
    assert layout["theme_filtered"] is True
    assert layout["kept_claim_ids"] == ["C0001", "C0002"]
    by_id = {node["id"]: node for node in layout["nodes"]}
    assert "C0003" not in by_id
    assert by_id["C0001"]["parent"] == "section:0"
    assert by_id["C0001"]["depth"] == 2
    assert by_id["C0002"]["parent"] == "C0001"
    assert by_id["C0002"]["depth"] == 3
    assert by_id["section:0"]["parent"] == "root"


def test_document_import_fails_when_nothing_fits_the_theme(tmp_path: Path):
    report = {
        "findings_all": [
            {"claim_id": "C0001", "claim": "A bakery sold bread."},
            {"claim_id": "C0002", "claim": "The oven was new."},
        ]
    }
    (tmp_path / "report_data.json").write_text(json.dumps(report), encoding="utf-8")

    class DropAll:
        def complete(self, prompt: str, **_kw: object) -> str:
            if "Keep claims that bear on one theme" in prompt:
                return json.dumps({"keep": [], "drop": ["C0001", "C0002"]})
            raise AssertionError("layout must not run when the theme gate drops every claim")

    with pytest.raises(RuntimeError, match="No claims fit the import theme"):
        apply_document_graph(
            tmp_path,
            auto_label=False,
            client=DropAll(),
            document_import=True,
            theme="Acme",
        )


def test_document_import_attach_does_not_reparent_existing_nodes(tmp_path: Path):
    report = {
        "topic": "Acme",
        "findings_all": [
            {"claim_id": "C0001", "claim": "The Cayman office opened in March."},
            {"claim_id": "C0002", "claim": "Acme hired a treasurer."},
            {"claim_id": "C0003", "claim": "Payroll ran through that treasurer."},
        ],
    }
    (tmp_path / "report_data.json").write_text(json.dumps(report), encoding="utf-8")
    existing = [
        {"claim_id": "section_bank", "claim": "Banking", "section": True, "depth": 1},
        {"claim_id": "claim_office", "claim": "Acme opened an office in Cayman.", "section": False, "depth": 2},
    ]

    class Scripted:
        def complete(self, prompt: str, **_kw: object) -> str:
            if "Keep claims that bear on one theme" in prompt:
                return json.dumps({"keep": ["C0001", "C0002", "C0003"], "drop": []})
            if "Existing map nodes" in prompt:
                return json.dumps(
                    {"parents": [{"claim_id": "C0001", "parent_claim_id": "claim_office"}]}
                )
            if "Lay out claims" in prompt:
                assert "Banking" in prompt
                return json.dumps(
                    {
                        "sections": [{"title": "Treasury", "claim_ids": ["C0002", "C0003"]}],
                        "parents": [{"claim_id": "C0003", "parent_claim_id": "C0002"}],
                    }
                )
            raise AssertionError(prompt[:80])

    apply_document_graph(
        tmp_path,
        auto_label=False,
        client=Scripted(),
        document_import=True,
        theme="Acme",
        attach=True,
        existing_nodes=existing,
    )
    layout = json.loads((tmp_path / "tree_layout.json").read_text(encoding="utf-8"))
    by_id = {node["id"]: node for node in layout["nodes"]}
    assert "claim_office" not in by_id
    assert "section_bank" not in by_id
    assert by_id["C0001"]["parent"] == "claim_office"
    assert by_id["C0002"]["parent"] == "section:0"
    assert by_id["C0003"]["parent"] == "C0002"
    assert by_id["C0003"]["depth"] == 3


def test_filter_and_layout_helpers_fail_open_on_bad_json():
    claims = [
        {"claim_id": "C0001", "claim": "Acme exists."},
        {"claim_id": "C0002", "claim": "Acme hired a treasurer."},
    ]
    boom = type("Boom", (), {"complete": staticmethod(lambda _prompt, **_kw: "{")})()
    assert filter_claims_for_theme(claims, "Acme", client=boom) is None
    sections, parents = layout_document_claims(claims, "Acme", client=boom)
    assert parents == {}
    assert sections == [{"title": "Acme", "claim_ids": ["C0001", "C0002"]}]


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
