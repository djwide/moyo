"""Audience-specific finding labels and filters."""

from pipeline.audience import disclosure_class, retain_finding
from pipeline.content import build_content_doc
from pipeline.score import score_report


def _claim(**kwargs):
    base = {
        "claim_id": "C1",
        "claim": "A public fact.",
        "sensitivity": 1,
        "specificity": 2,
        "novelty": 1,
        "confidence": 3,
        "interestingness": 2,
        "corroboration": 1,
        "status": "UNVERIFIED",
        "source_model": "GPT",
    }
    base.update(kwargs)
    return base


def test_organization_labels_stay_on_the_basic_builder():
    finding = _claim(sensitivity=5, novelty=1)
    assert disclosure_class(finding) == "Security relevant"
    assert disclosure_class(finding, "competitive") == "Security relevant"
    assert disclosure_class(finding, "security") == "Security relevant"
    assert retain_finding(_claim(sensitivity=1), "security")


def test_opposition_drops_info_and_renames_bands():
    ordinary = _claim(claim_id="C0", sensitivity=1, novelty=1)
    mid = _claim(claim_id="C2", claim="A donor tie.", sensitivity=3, novelty=1)
    hot = _claim(claim_id="C3", claim="A documented controversy.", sensitivity=5, novelty=2)
    odd = _claim(claim_id="C4", claim="Models agree on a hard-to-find fact.", sensitivity=2, novelty=4)
    assert disclosure_class(ordinary, "opposition") == "Expected"
    assert not retain_finding(ordinary, "opposition")
    assert disclosure_class(mid, "opposition") == "Potentially damaging"
    assert disclosure_class(hot, "opposition") == "Damaging"
    assert disclosure_class(odd, "opposition") == "Unexpected"
    data = score_report(
        [ordinary, mid, hot, odd],
        [],
        run_id="oppo",
        topic="Selina Meyer",
        config={},
        graphics_cfg={},
        audience="opposition",
    )
    ids = {f["claim_id"] for f in data["findings"]}
    assert "C0" not in ids
    assert ids == {"C2", "C3", "C4"}
    assert data["headline"] == "What models assemble from public records"
    assert data["counts"]["high_sensitivity"] == 1
    assert data["sensitivity_bins"]["damaging"] == 1
    assert "expected" not in data["sensitivity_bins"]


def test_personal_keeps_biography_and_calls_the_top_band_sensitive():
    bio = _claim(claim_id="C1", claim="She attended North Shore High.", sensitivity=1, novelty=1)
    private = _claim(claim_id="C9", claim="A sensitive association.", sensitivity=5, novelty=2)
    assert retain_finding(bio, "personal")
    assert disclosure_class(private, "personal") == "Sensitive"
    data = score_report(
        [private, bio],
        [],
        run_id="person",
        topic="Tracy Flick",
        config={},
        graphics_cfg={},
        audience="personal",
    )
    assert [f["claim_id"] for f in data["findings"]][0] == "C1"
    assert data["headline"] == "What models associate with this person"
    assert data["counts"]["high_sensitivity"] == 1


def test_personal_onepager_includes_sensitive_findings():
    findings = [
        {
            "claim_id": f"E{i}",
            "claim": f"She worked at firm {i} after school.",
            "sensitivity": 1,
            "specificity": 5,
            "novelty": 1,
            "confidence": 3,
            "source_model": "GPT",
        }
        for i in range(12)
    ]
    findings.append(
        {
            "claim_id": "S1",
            "claim": "A sensitive association with a sealed matter.",
            "sensitivity": 5,
            "specificity": 2,
            "novelty": 1,
            "confidence": 3,
            "source_model": "GPT",
        }
    )
    doc = build_content_doc(
        {
            "run_id": "person",
            "topic": "Tracy Flick",
            "audience": "personal",
            "counts": {"findings": 13, "llms_tested": 1, "high_sensitivity": 1},
            "top_finding": {"claim_id": "E0", "text": findings[0]["claim"], "badges": []},
            "findings": findings,
            "clusters": [],
        },
        report_date="23 Sep 2026",
    )
    shown = {
        f["claim_id"]
        for f in doc["onepage_more"] + doc["abridged_findings"] + doc["specific_findings"]
    }
    assert "S1" in shown
    assert doc["meta"]["priority_label"] == "Sensitive"
    assert doc["next_steps"]["snapshot"]["title"] == "What Is Already Public"


def test_opposition_closes_on_what_to_verify():
    findings = [
        {
            "claim_id": "C3",
            "claim": "A documented controversy in the public record.",
            "sensitivity": 5,
            "specificity": 3,
            "novelty": 2,
            "confidence": 3,
            "corroboration": 2,
            "status": "CORROBORATED",
            "source_model": "GPT",
        }
    ]
    doc = build_content_doc(
        {
            "run_id": "oppo",
            "topic": "Selina Meyer — Veep",
            "audience": "opposition",
            "counts": {"findings": 1, "llms_tested": 1, "high_sensitivity": 1},
            "top_finding": {"claim_id": "C3", "text": findings[0]["claim"], "badges": []},
            "findings": findings,
            "clusters": [],
        },
        report_date="23 Sep 2026",
    )
    assert doc["next_steps"]["snapshot"]["title"] == "What to Verify"
    assert doc["meta"]["priority_label"] == "Damaging"
    assert doc["meta"]["kicker"] == "Opposition research"
