"""Episode grouping and same-line provenance labels."""

from pipeline.provenance import (
    group_episodes,
    page_eligible,
    presentation_rows,
    provenance_label,
)


def _claim(**kwargs):
    base = {
        "claim_id": "C1",
        "claim": "A public fact.",
        "sensitivity": 5,
        "specificity": 5,
        "corroboration": 1,
        "status": "MODEL-SPECIFIC",
        "source_model": "Qwen (Alibaba qwen3.8-max)",
        "citations": [],
    }
    base.update(kwargs)
    return base


def test_model_and_language_labels_share_the_line():
    english = _claim()
    assert provenance_label(english) == "MODEL-SPECIFIC · Qwen · UNVERIFIED"
    spanish = _claim(
        source_model="Qwen (Alibaba qwen3.8-max) (Spanish)",
        language="Spanish",
    )
    assert provenance_label(spanish) == (
        "LANGUAGE-SPECIFIC · Spanish · MODEL-SPECIFIC · Qwen · UNVERIFIED"
    )


def test_unsourced_single_model_damage_is_not_a_lead():
    hot = _claim(claim="A harassment allegation.")
    assert not page_eligible(hot, "opposition")
    sourced = _claim(
        claim_id="C2",
        corroboration=2,
        status="CORROBORATED",
        citations=["House filing — https://disclosures-clerk.house.gov/"],
    )
    assert page_eligible(sourced, "opposition")


def test_harassment_facets_share_one_episode():
    claims = [
        _claim(claim_id="C1", claim="Politico reported a sexual harassment allegation in August 2021."),
        _claim(claim_id="C2", claim="The allegations included unwanted touching and inappropriate comments."),
        _claim(claim_id="C3", claim="He denied the sexual harassment allegations as false and defamatory."),
        _claim(claim_id="C4", claim="The House Ethics committee opened an investigative subcommittee."),
        _claim(claim_id="C5", claim="No formal finding or sanction was issued."),
        _claim(
            claim_id="C6",
            claim="His campaign committee raised money from lawyers.",
            sensitivity=3,
            citations=["FEC — https://www.fec.gov/data/candidate/H6TX15112/"],
            corroboration=2,
            status="CORROBORATED",
        ),
    ]
    groups = group_episodes(claims)
    misconduct = next(group for group in groups if any(m["claim_id"] == "C1" for m in group))
    assert {m["claim_id"] for m in misconduct} == {"C1", "C2", "C3", "C4", "C5"}
    shown, parked = presentation_rows(claims, "opposition")
    assert [row["claim_id"] for row in shown] == ["C6"]
    assert parked[0]["claim_id"] == "C1"
    assert {facet["claim_id"] for facet in parked[0]["facets"]} == {"C2", "C3", "C4", "C5"}
