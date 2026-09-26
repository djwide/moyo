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


def test_chunk_bibliography_is_not_inherited():
    citations = resolve_claim_citations(
        claim="In August 2021, Politico reported a harassment allegation.",
        excerpt="In August 2021, Politico published a report detailing allegations.",
        llm_citations=[
            '"Former aide accuses Texas Rep. Vicente Gonzalez of sexual harassment," Aug. 2021'
        ],
        chunk_citations=[
            "House.gov official biography",
            "Biographical Directory of the United States Congress",
            "https://clerk.house.gov/",
        ],
    )
    assert citations == []


def test_stored_url_absent_from_excerpt_is_dropped():
    citations = resolve_claim_citations(
        claim="He was criticized for a comment about his rival.",
        excerpt="In 2022 he was criticized for a comment about his rival Mayra Flores.",
        llm_citations=["https://www.fec.gov/data/candidate/H6TX15112/"],
        chunk_citations=["https://disclosures-clerk.house.gov/"],
    )
    assert citations == []


def test_footer_url_matches_excerpt_by_name():
    citations = resolve_claim_citations(
        claim="OpenSecrets lists his top donors.",
        excerpt="OpenSecrets lists his top donors for the 2024 cycle.",
        chunk_citations=[
            "https://www.opensecrets.org/members-of-congress/vicente-gonzalez/summary",
            "https://clerk.house.gov/",
            "https://www.fec.gov/data/candidate/H6TX15112/",
        ],
    )
    assert citations == [
        "https://www.opensecrets.org/members-of-congress/vicente-gonzalez/summary"
    ]


def test_named_outlet_without_matching_url_stays_unverified():
    citations = resolve_claim_citations(
        claim="Politico reported a harassment allegation.",
        excerpt="In August 2021, Politico published a report detailing allegations.",
        chunk_citations=[
            "https://clerk.house.gov/",
            "https://www.fec.gov/data/candidate/H6TX15112/",
        ],
    )
    assert citations == []


def test_numbered_marker_still_resolves_when_excerpt_contains_it():
    citations = resolve_claim_citations(
        claim="The disclosure lists outside income.",
        excerpt="The disclosure lists outside income.[3]",
        llm_citations=[],
        chunk_citations=[
            "https://clerk.house.gov/",
            "https://www.opensecrets.org/members-of-congress/summary",
        ],
        reference_map={
            "3": "House Clerk — https://disclosures-clerk.house.gov/public_disc/financial-pdfs/2024/10056131.pdf"
        },
    )
    assert citations == [
        "House Clerk — https://disclosures-clerk.house.gov/public_disc/financial-pdfs/2024/10056131.pdf"
    ]


def test_excerpt_url_is_kept():
    citations = resolve_claim_citations(
        claim="The filing is public.",
        excerpt="See the filing at https://disclosures-clerk.house.gov/.",
        llm_citations=["Politico"],
        chunk_citations=["https://www.fec.gov/"],
    )
    assert citations == ["https://disclosures-clerk.house.gov/"]


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


def test_title_surfaces_use_only_report_date():
    """Cover / one-pager / exec stamp must not also print generated_at."""
    root = Path(__file__).resolve().parents[1] / "reports" / "design-system"
    dated = [
        root / "templates" / "onepage.html.j2",
        root / "pages" / "cover.j2",
    ]
    undated = [root / "pages" / "executive_summary.j2"]
    for path in dated + undated:
        text = path.read_text(encoding="utf-8")
        assert "generated_at" not in text, path.name
        assert "format_timestamp" not in text, path.name
        if path in dated:
            assert "report_date" in text, path.name
