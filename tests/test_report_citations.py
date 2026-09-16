from pathlib import Path

from pipeline.citations import (
    citation_entry,
    extract_reference_map,
    resolve_claim_citations,
)
from pipeline.sources import build_source_registry


def test_numeric_llm_citation_resolves_to_reference_text():
    answer = """
The filing confirms the acquisition.[18]

## Sources
18. Example filing — https://example.com/filing
"""
    refs = extract_reference_map(answer)

    citations = resolve_claim_citations(
        claim="The filing confirms the acquisition.",
        excerpt="The filing confirms the acquisition.[18]",
        llm_citations=["18"],
        chunk_citations=["Example filing — https://example.com/filing"],
        reference_map=refs,
    )

    assert citations == ["Example filing — https://example.com/filing"]


def test_unresolved_reference_number_never_becomes_a_source():
    assert citation_entry("[18]")["label"] == ""
    assert citation_entry("[18] https://example.com/filing") == {
        "label": "example.com",
        "url": "https://example.com/filing",
        "text": "[18] https://example.com/filing",
    }

    sources, findings = build_source_registry(
        [
            {
                "claim_id": "C1",
                "citations": ["18", "Example filing — https://example.com/filing"],
            }
        ]
    )

    assert [source["label"] for source in sources] == ["Example filing"]
    assert findings[0]["source_refs"] == ["S1"]


def test_onepage_template_contains_no_anchor_elements():
    template = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "design-system"
        / "templates"
        / "onepage.html.j2"
    ).read_text(encoding="utf-8")

    assert "<a " not in template.lower()
    assert "<a>" not in template.lower()
