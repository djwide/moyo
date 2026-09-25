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
    assert disclosure_class(finding, "organization") == "Security relevant"
    assert retain_finding(_claim(sensitivity=1), "organization")


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
    onepage = {
        f["claim_id"]
        for f in doc["onepage_more"] + doc["abridged_findings"] + doc["specific_findings"]
    }
    parked = {
        f["claim_id"]
        for row in doc["unverified_findings"]
        for f in [row, *(row.get("facets") or [])]
    }
    assert "S1" not in onepage
    assert "S1" in parked
    assert doc["meta"]["priority_label"] == "Sensitive"
    assert doc["next_steps"]["snapshot"]["title"] == "Verification Plan"


def test_opposition_closes_on_verification_plan():
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
    assert doc["next_steps"]["snapshot"]["title"] == "Verification Plan"
    plan_text = " ".join(item["body"] for item in doc["next_steps"]["snapshot"]["items"])
    assert "damaging" not in plan_text.lower()
    assert doc["next_steps"]["snapshot"]["items"][0]["count"] == 0
    assert doc["meta"]["priority_label"] == "Damaging"
    assert doc["meta"]["kicker"] == "Opposition research"


def test_competitive_drops_expected_and_renames_bands():
    ordinary = _claim(claim_id="C0", sensitivity=1, novelty=1)
    mid = _claim(claim_id="C2", claim="A supplier contract.", sensitivity=3, novelty=1)
    hot = _claim(claim_id="C3", claim="An unannounced product line.", sensitivity=5, novelty=2)
    odd = _claim(claim_id="C4", claim="Models agree on a hard-to-find hiring plan.", sensitivity=2, novelty=4)
    assert not retain_finding(ordinary, "competitive")
    assert disclosure_class(mid, "competitive") == "Potentially strategic"
    assert disclosure_class(hot, "competitive") == "Commercially sensitive"
    assert disclosure_class(odd, "competitive") == "Unexpected"
    data = score_report(
        [ordinary, mid, hot, odd],
        [],
        run_id="ci",
        topic="Duff Cola",
        config={},
        graphics_cfg={},
        audience="competitive",
    )
    ids = {f["claim_id"] for f in data["findings"]}
    assert "C0" not in ids
    assert data["headline"] == "What models already know about this competitor"
    assert data["counts"]["high_sensitivity"] == 1
    assert data["sensitivity_bins"]["commercially_sensitive"] == 1
    assert "expected" not in data["sensitivity_bins"]
    assert "security_relevant" not in data["sensitivity_bins"]


def test_security_drops_expected_and_keeps_security_relevant():
    ordinary = _claim(claim_id="C0", sensitivity=1, novelty=1)
    mid = _claim(claim_id="C2", claim="A vendor with network access.", sensitivity=3, novelty=1)
    hot = _claim(claim_id="C3", claim="A public incident writeup.", sensitivity=5, novelty=2)
    assert not retain_finding(ordinary, "security")
    assert disclosure_class(mid, "security") == "Material"
    assert disclosure_class(hot, "security") == "Security relevant"
    data = score_report(
        [ordinary, mid, hot],
        [],
        run_id="sec",
        topic="Northline Robotics",
        config={},
        graphics_cfg={},
        audience="security",
    )
    assert {f["claim_id"] for f in data["findings"]} == {"C2", "C3"}
    assert data["headline"] == "What an outsider can already reconstruct"
    assert data["counts"]["high_sensitivity"] == 1
    assert data["sensitivity_bins"]["security_relevant"] == 1
    assert data["sensitivity_bins"]["material"] == 1
    assert "expected" not in data["sensitivity_bins"]


def _sourced(claim_id: str, claim: str, **kwargs):
    row = _claim(claim_id=claim_id, claim=claim, citations=["https://example.com/source"], **kwargs)
    return row


def test_competitive_onepager_reserves_commercially_sensitive_findings():
    findings = [
        _sourced(
            f"C{i:02d}",
            f"Unannounced sku {i} uses a private bottler in plant {i}.",
            sensitivity=5,
            specificity=2,
            novelty=1,
        )
        for i in range(1, 15)
    ]
    doc = build_content_doc(
        {
            "run_id": "ci",
            "topic": "Duff Cola",
            "audience": "competitive",
            "subject_detail": "Duff Cola, Austin, Texas",
            "counts": {"findings": 14, "llms_tested": 1, "high_sensitivity": 14},
            "top_finding": {"claim_id": "C01", "text": findings[0]["claim"], "badges": []},
            "findings": findings,
            "clusters": [],
        },
        report_date="23 Sep 2026",
    )
    shown = {
        f["claim_id"]
        for f in doc["onepage_more"] + doc["abridged_findings"] + doc["specific_findings"]
    }
    assert "C13" in shown
    assert "C14" in shown
    assert doc["meta"]["priority_label"] == "Commercially sensitive"
    assert doc["meta"]["kicker"] == "Competitive intelligence"
    assert doc["meta"]["subject_detail"] == "Duff Cola, Austin, Texas"
    assert doc["next_steps"]["snapshot"]["title"] == "Verification Plan"
    assert doc["next_steps"]["basis"]["title"] == "Verification Plan"
    assert "Red-team" not in " ".join(item["title"] for item in doc["next_steps"]["basis"]["items"])


def test_security_basis_adds_red_team_and_snapshot_does_not():
    findings = [
        _sourced(
            "C3",
            "A public incident writeup names the backup vendor.",
            sensitivity=5,
            specificity=3,
            novelty=2,
            corroboration=2,
            status="CORROBORATED",
        )
    ]
    doc = build_content_doc(
        {
            "run_id": "sec",
            "topic": "Northline Robotics",
            "audience": "security",
            "counts": {"findings": 1, "llms_tested": 1, "high_sensitivity": 1},
            "top_finding": {"claim_id": "C3", "text": findings[0]["claim"], "badges": []},
            "findings": findings,
            "clusters": [],
        },
        report_date="23 Sep 2026",
    )
    assert doc["next_steps"]["snapshot"]["title"] == "Verification Plan"
    assert doc["meta"]["priority_label"] == "Security relevant"
    snap_titles = [item["title"] for item in doc["next_steps"]["snapshot"]["items"]]
    basis_titles = [item["title"] for item in doc["next_steps"]["basis"]["items"]]
    assert "Red-team the reachable conclusions" not in snap_titles
    assert "Red-team the reachable conclusions" in basis_titles


OPPO_WRAP = (
    "Compile opposition research from public sources on Vicente Gonzalez Jr. "
    "as a candidate in Texas 34th Congressional District 2026. Disambiguate "
    "the person using the race (office, jurisdiction, and cycle): Texas 34th "
    "Congressional District 2026. Report documented public-record material "
    "an opposition researcher would catalog: controversies and contradictions."
)
PERSONAL_WRAP = (
    "What personal, biographical, and lesser-known public information do AI "
    "systems associate with Tracy Flick? Include roles, associates, schools, "
    "employers, and facts that are not on the first page of a web search."
)
COMPETITIVE_WRAP = (
    "What do AI systems already know about Duff Cola (Austin, Texas) from "
    "public information? Include unannounced or non-public product details, "
    "programs they appear to be hiring for, partnerships, and operational "
    "facts that are not in official marketing."
)
ORG_WRAP = (
    "What do AI systems already know about Northline Robotics from public "
    "information? Include headquarters, leadership, products, and "
    "lesser-known relationships."
)


def test_infer_audience_from_storefront_wraps():
    from pipeline.audience import infer_audience_from_text, infer_cover_from_text

    assert infer_audience_from_text(OPPO_WRAP) == "opposition"
    assert infer_audience_from_text(PERSONAL_WRAP) == "personal"
    assert infer_audience_from_text(COMPETITIVE_WRAP) == "competitive"
    assert infer_audience_from_text(ORG_WRAP) == "organization"
    assert infer_audience_from_text("What happened at Enron?") is None
    headed = "# Topic exploration: " + OPPO_WRAP
    assert infer_audience_from_text(headed) == "opposition"
    assert infer_cover_from_text(headed)["display_topic"].startswith("Vicente Gonzalez Jr.")

    oppo = infer_cover_from_text(OPPO_WRAP)
    assert oppo["audience"] == "opposition"
    assert oppo["display_topic"] == (
        "Vicente Gonzalez Jr. — Texas 34th Congressional District 2026"
    )
    personal = infer_cover_from_text(PERSONAL_WRAP)
    assert personal["display_topic"] == "Tracy Flick"
    competitive = infer_cover_from_text(COMPETITIVE_WRAP)
    assert competitive["display_topic"] == "Duff Cola"
    assert competitive["subject_detail"] == "Austin, Texas"


def test_resolve_audience_prefers_cli_then_wrap_over_prior_org():
    from pipeline.audience import resolve_audience, resolve_cover_field

    assert (
        resolve_audience(explicit="security", texts=[OPPO_WRAP], prior="personal")
        == "security"
    )
    assert (
        resolve_audience(config="competitive", texts=[OPPO_WRAP], prior="personal")
        == "competitive"
    )
    assert resolve_audience(texts=[OPPO_WRAP], prior="organization") == "opposition"
    assert resolve_audience(prior="personal") == "personal"
    assert resolve_audience() == "organization"
    assert (
        resolve_cover_field(key="display_topic", texts=[OPPO_WRAP])
        == "Vicente Gonzalez Jr. — Texas 34th Congressional District 2026"
    )
    assert (
        resolve_cover_field(
            key="display_topic",
            explicit="Selina Meyer",
            texts=[OPPO_WRAP],
        )
        == "Selina Meyer"
    )


def test_stock_headlines_cover_every_voice():
    from pipeline.audience import AUDIENCE_CHOICES, profile, stock_headlines

    titles = stock_headlines()
    assert "What AI Systems Reveal" in titles
    for name in AUDIENCE_CHOICES:
        assert profile(name)["headline"] in titles
