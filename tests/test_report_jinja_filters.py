from pipeline.jinja_filters import (
    clip,
    display_status,
    format_int,
    format_number,
    format_score,
    format_timestamp,
    link_cites,
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
                "models_tested": [
                    "ChatGPT (OpenAI gpt-4o)",
                    "Claude (Anthropic Sonnet)",
                    "ChatGPT (OpenAI gpt-4o) (French)",
                ],
                "strategies": ["paraphrase"],
            },
        },
        report_date="18 Sep 2026",
        aliases={
            "ChatGPT (OpenAI gpt-4o)": "GPT",
            "Claude (Anthropic Sonnet)": "Claude",
        },
    )
    assert doc["meta"]["models_tested"] == [
        "ChatGPT (OpenAI gpt-4o)",
        "Claude (Anthropic Sonnet)",
    ]
    assert "GPT" not in doc["meta"]["models_tested"]
    assert doc["pages"]["executive_summary"]["title"] == "Disclosure Summary"
    assert doc["pages"]["findings"]["title"] == "Priority Findings"
    assert doc["pages"]["risk_overview"]["title"] == "Model Exposure"
    assert doc["pages"]["sources"]["title"] == "Cited Sources"
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
            "models_tested": ["ChatGPT (OpenAI gpt-4o)"],
            "strategies": ["original"],
            "include_remediation": False,
        },
        "pages": {
            "executive_summary": {
                "title": "Disclosure Summary",
                "body": "The formula appears in S21.",
            },
            "risk_overview": {"title": "Model Exposure"},
            "findings": {"title": "Priority Findings"},
            "evidence": {"title": "Verbatim Excerpts"},
            "model_comparison": {"title": "Model Comparison"},
            "appendix": {
                "title": "Appendix",
                "claims_title": "Finding Index",
                "method_title": "Collection Method",
                "corpus_title": "Normalized Responses",
            },
            "inventory": {"title": "Cluster Inventory"},
            "sources": {"title": "Cited Sources"},
            "glossary": {"title": "Score Glossary"},
        },
        "assets": {},
        "abridged_findings": [
            {
                "claim_id": "C0001",
                "claim": "See S21 for the named source.",
                "status": "UNVERIFIED",
                    "sensitivity": 3,
                    "specificity": 3,
                "source_refs": ["S21"],
                "citations_display": [
                    {"ref": "S21", "label": "The Coca-Cola Company"}
                ],
            }
        ],
        "evidence_findings": [],
        "top_finding": {"text": "x", "badges": []},
        "specific_findings": [],
        "next_steps": {
            "snapshot": {"title": "Next", "items": []},
            "basis": {"title": "Next", "items": []},
        },
        "basis": {
            "inventory": [],
            "findings_full": [],
            "chain_details": [
                {
                    "label": "Empty cite chain",
                    "chain_id": "CH1",
                    "model_count": 1,
                    "score": 1,
                    "recovered": "A claim without sources.",
                    "citations": [{"ref": "", "label": ""}],
                    "derivation": {"steps": []},
                    "corroborating_outputs": [],
                    "implication": "",
                }
            ],
        },
        "model_contrast": {},
        "sources": [
            {
                "ref": "S21",
                "label": "The Coca-Cola Company",
                "cited_by": 1,
                "url": "https://example.test",
            }
        ],
        "glossary": [],
        "model_dossiers": [
            {
                "slug": "gpt",
                "model": "GPT",
                "id": "model-gpt",
                "detail_id": "model-gpt-detail",
                "lede": "GPT produced 1 exclusive finding.",
                "findings": 1,
                "unique": 1,
                "shared": 0,
                "high": 1,
                "overlap_pct": 0,
                "exclusive_citation_count": 0,
                "probes": {"answered": 1, "attempted": 1, "failed": 0, "empty": 0},
                "distinctive": [],
                "unique_findings": [],
                "exclusive_citations": [],
                "failed_probes": [],
                "charts": {
                    "fingerprint": "d_gpt_fingerprint",
                    "mix": "d_gpt_mix",
                    "probes": "d_gpt_probes",
                    "overlap": "d_gpt_overlap",
                },
            }
        ],
        "followups": [],
        "response_corpus": [
            {
                "model": "GPT",
                "query_id": "q1",
                "query": "how?",
                "text": "SNAPSHOT_MUST_OMIT_THIS_CORPUS",
                "failed": False,
            }
        ],
        "corpus_groups": [
            {
                "model": "GPT",
                "items": [
                    {
                        "model": "GPT",
                        "query_id": "q1",
                        "query": "how?",
                        "text": "SNAPSHOT_MUST_OMIT_THIS_CORPUS",
                        "failed": False,
                    }
                ],
            }
        ],
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
    assert "Disclosure Summary" in snap
    assert "Model Exposure" in snap
    assert "Verbatim Excerpts" in snap
    assert "Priority Findings" in snap
    assert "Model Comparison" in snap
    assert "Finding Index" in snap
    assert "Cited Sources" in snap
    assert "Score Glossary" in snap
    assert "Cluster Inventory" in basis
    assert "Cluster Transcripts" in basis
    assert "Exposure Chains" in basis
    assert "Appendix" in basis
    assert "Collection Method" in basis
    assert "Normalized Responses" not in basis
    assert 'id="appendix-method"' in basis
    assert 'id="appendix-responses"' not in basis
    assert "SNAPSHOT_MUST_OMIT_THIS_CORPUS" not in basis
    assert "SNAPSHOT_MUST_OMIT_THIS_CORPUS" not in snap
    assert "Normalized Responses" not in snap
    assert 'id="appendix-responses"' not in snap
    assert "The model provided no citations for this assertion." in basis
    assert "toc--sub" in basis
    assert 'id="model-gpt"' in snap
    assert 'id="model-gpt"' in basis
    assert 'id="model-gpt-detail"' in basis
    assert 'id="model-gpt-detail"' not in snap
    assert "Score Fingerprint" in snap
    assert "Exclusive vs Shared" in basis
    assert "model-exposure-list" not in snap
    assert 'id="cite-S21"' in snap
    assert 'href="#cite-S21"' in snap
    assert 'id="cite-S21"' in basis
    assert 'href="#cite-S21"' in basis
    onepage = env.get_template("onepage.html.j2").render(
        content=content,
        graphics={},
        logo_uri="",
        partner_logo_uri="",
        favicon_uri="",
        css_href="css/onepage.css",
    )
    assert "ChatGPT (OpenAI gpt-4o)" in snap
    assert "ChatGPT (OpenAI gpt-4o)" in basis
    assert "ChatGPT (OpenAI gpt-4o)" in onepage
    assert "models-line" in onepage


def test_link_cites_wraps_source_refs_and_escapes_html():
    html = str(link_cites('Named in S21 and S3, not <script>alert(1)</script>.'))
    assert '<a class="cite-ref" href="#cite-S21">S21</a>' in html
    assert '<a class="cite-ref" href="#cite-S3">S3</a>' in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_group_response_corpus_preserves_model_order():
    from pipeline.content import group_response_corpus

    groups = group_response_corpus(
        [
            {"model": "Claude", "query_id": "a"},
            {"model": "GPT", "query_id": "b"},
            {"model": "Claude", "query_id": "c"},
        ]
    )
    assert [g["model"] for g in groups] == ["Claude", "GPT"]
    assert [row["query_id"] for row in groups[0]["items"]] == ["a", "c"]


def test_snapshot_markdown_omits_normalized_responses():
    from pipeline.content import render_report_md

    md = render_report_md(
        {
            "meta": {
                "topic": "Vault",
                "run_id": "t",
                "report_date": "21 Sep 2026",
                "counts": {},
            },
            "pages": {
                "executive_summary": {"title": "Disclosure Summary", "body": "x"},
                "risk_overview": {"title": "Model Exposure", "body": ""},
                "model_comparison": {"title": "Model Comparison", "body": ""},
                "findings": {"title": "Priority Findings", "body": ""},
                "appendix": {"corpus_title": "Normalized Responses"},
            },
            "next_steps": {"snapshot": {"items": []}},
            "response_corpus": [
                {"model": "GPT", "query_id": "q1", "query": "q", "text": "secret"}
            ],
        }
    )
    assert "Normalized Responses" not in md
    assert "secret" not in md


def test_build_next_steps_mentions_isvf_on_basis_follow_up():
    from pipeline.content import build_next_steps

    steps = build_next_steps(include_remediation=False)
    titles = [item["title"] for item in steps["basis"]["items"]]
    assert "Verify organizational policy" in titles
    isvf = next(
        item
        for item in steps["basis"]["items"]
        if item["title"] == "Verify organizational policy"
    )
    assert "Idea Security Verification Framework" in isvf["body"]
    assert "information security policy" in isvf["body"]
