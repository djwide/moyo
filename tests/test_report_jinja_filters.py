from pipeline.jinja_filters import (
    clip,
    display_status,
    format_int,
    format_number,
    format_score,
    format_timestamp,
)
from pipeline.content import build_content_doc


def test_format_int_adds_thousands_separators():
    assert format_int(12450.432) == "12,450"
    assert format_int(None) == "0"


def test_format_number_and_score():
    assert format_number(3) == "3"
    assert format_number(3.26, 1) == "3.3"
    assert format_score(4) == "4/5"


def test_display_status_title_cases_labels():
    assert display_status("UNVERIFIED") == "Unverified"
    assert display_status("model-specific") == "Model-specific"
    assert display_status("high") == "High"
    assert display_status("SPECIFIC") == "Specific"


def test_clip_trims_on_word_boundary():
    text = "The gist publishes the Vault path prod/secrets/db-root for HALCYON Postgres."
    out = clip(text, 40)
    assert len(out) <= 41
    assert out.endswith("…")
    assert "Postgres" not in out


def test_format_timestamp_humanizes_iso():
    assert "18 Sep 2026" in format_timestamp("2026-09-18T14:03:00Z")


def test_build_content_doc_uses_action_titles():
    doc = build_content_doc(
        {
            "run_id": "t",
            "topic": "HALCYON credentials",
            "headline": "What AI Systems Reveal",
            "generated_at": "2026-09-18T14:03:00Z",
            "counts": {
                "findings": 12,
                "llms_tested": 4,
                "high_sensitivity": 3,
                "chains": 2,
            },
            "top_finding": {
                "claim_id": "C0001",
                "text": "Vault path prod/secrets/db-root is public.",
                "badges": ["HIGH"],
            },
            "findings": [
                {
                    "claim_id": "C0001",
                    "claim": "Vault path prod/secrets/db-root is public.",
                    "status": "CORROBORATED",
                    "sensitivity": 5,
                    "specificity": 5,
                    "source_model": "ChatGPT",
                    "confidence": 4,
                    "raw_excerpt": "prod/secrets/db-root",
                    "raw_start_line": 42,
                    "raw_end_line": 44,
                }
            ],
            "clusters": [],
            "explore_meta": {
                "models_tested": ["ChatGPT", "Claude"],
                "strategies": ["paraphrase"],
            },
        },
        report_date="18 Sep 2026",
    )
    assert "high-sensitivity" in doc["pages"]["executive_summary"]["title"]
    assert doc["pages"]["findings"]["title"] == "1 finding that carries this exposure"
    assert doc["pages"]["risk_overview"]["title"] == "Which models disclosed the most"
    assert doc["pages"]["sources"]["title"] == "What the models cited"
    assert "Overview" not in doc["pages"]["executive_summary"]["title"]
    assert "reputational and compliance risk" not in (
        doc["pages"]["executive_summary"].get("why_it_matters") or ""
    )


def test_snapshot_keeps_more_than_five_findings():
    findings = [
        {
            "claim_id": f"C{i:04d}",
            "claim": f"Fact number {i} about the vault path.",
            "status": "UNVERIFIED",
            "sensitivity": 4 if i < 8 else 2,
            "specificity": 5 if i < 6 else 3,
            "source_model": "ChatGPT",
            "confidence": 3,
        }
        for i in range(1, 16)
    ]
    doc = build_content_doc(
        {
            "run_id": "t",
            "topic": "Vault",
            "counts": {"findings": 15, "llms_tested": 1, "high_sensitivity": 7},
            "top_finding": {"claim_id": "C0001", "text": findings[0]["claim"], "badges": []},
            "findings": findings,
            "clusters": [],
        },
        report_date="20 Sep 2026",
    )
    assert len(doc["abridged_findings"]) == 12
    assert len(doc["specific_findings"]) == 4
    assert len(doc["onepage_more"]) <= 6
    assert doc["onepage_more"]
    assert len(doc["basis"]["findings_full"]) == 15
    listed = {f["claim_id"] for f in doc["abridged_findings"]}
    for f in doc["evidence_findings"]:
        assert f["claim_id"] in listed
    assert len(doc["evidence_findings"]) <= 10


def test_snapshot_and_basis_templates_include_table_of_contents():
    from pathlib import Path

    from jinja2 import Environment, FileSystemLoader, select_autoescape

    from pipeline.jinja_filters import register_filters

    ds = Path(__file__).resolve().parents[1] / "reports" / "design-system"
    env = Environment(
        loader=FileSystemLoader([str(ds / "templates"), str(ds)]),
        autoescape=select_autoescape(["html", "xml"]),
    )
    register_filters(env)
    env.filters["md"] = lambda value: value or ""
    env.filters["plain"] = lambda value: value or ""
    content = {
        "meta": {
            "topic": "Vault",
            "prompts": ["vault"],
            "report_date": "21 Sep 2026",
            "counts": {
                "findings": 1,
                "llms_tested": 1,
                "llms_attempted": 1,
                "llms_response_received": 1,
                "llms_substantive": 1,
                "llms_claims_contributed": 1,
                "high_sensitivity": 0,
                "chains": 0,
                "languages": 0,
            },
            "coverage": {
                "attempted": 1,
                "response_received": 1,
                "substantive_response": 1,
                "claims_contributed": 1,
            },
            "models_tested": ["GPT"],
            "strategies": ["original"],
            "include_remediation": False,
        },
        "pages": {
            "executive_summary": {"title": "What the models disclosed"},
            "risk_overview": {"title": "Which models disclosed the most"},
            "findings": {"title": "Findings that carry this exposure"},
            "evidence": {"title": "Verbatim excerpts with line numbers"},
            "model_comparison": {"title": "Where the models validate"},
            "appendix": {"claims_title": "Cluster index"},
            "inventory": {"title": "Every cluster, ranked"},
            "sources": {"title": "What the models cited"},
            "glossary": {"title": "How to read the scores"},
        },
        "assets": {},
        "abridged_findings": [],
        "evidence_findings": [],
        "top_finding": {"text": "x", "badges": []},
        "specific_findings": [],
        "next_steps": {
            "snapshot": {"title": "Next", "items": []},
            "basis": {"title": "Next", "items": []},
        },
        "basis": {"inventory": [], "findings_full": [], "chain_details": []},
        "model_contrast": {},
        "sources": [],
        "glossary": [],
        "followups": [],
        "response_corpus": [],
    }
    snap = env.get_template("report.html.j2").render(
        content=content,
        graphics={},
        logo_uri="",
        partner_logo_uri="",
        favicon_uri="",
        css_href="css/report.css",
    )
    basis = env.get_template("basis.html.j2").render(
        content=content,
        graphics={},
        logo_uri="",
        partner_logo_uri="",
        favicon_uri="",
        css_href="css/basis.css",
    )
    assert 'class="page page--contents"' in snap
    assert 'class="page page--contents"' in basis
    assert 'href="#evidence"' in snap
    assert 'id="evidence"' in snap
    assert 'href="#inventory"' in basis
    assert 'id="inventory"' in basis
    assert "Contents" in snap
    assert "Contents" in basis
