from pipeline.overview import exposure_overview, model_rows, reproduction, verification_plan


def _f(cid, models, sens=4, url=False, status=""):
    return {
        "claim_id": cid,
        "claim": f"Claim {cid} about the subject.",
        "sensitivity": sens,
        "source_models": models,
        "source_model": models[0],
        "corroboration": len(models),
        "status": status,
        "citations": ["https://example.test/a"] if url else [],
    }


def test_model_rows_keep_findings_and_high_significance_separate():
    rows = model_rows(
        [_f("C1", ["GPT", "Claude"], sens=5, url=True), _f("C2", ["GPT"], sens=2)],
        roster=["GPT", "Claude", "Grok"],
    )
    by = {row["model"]: row for row in rows}
    assert by["GPT"]["findings"] == 2 and by["GPT"]["high"] == 1
    assert by["GPT"]["verified"] == 1 and by["GPT"]["corroborated"] == 1
    assert by["Grok"]["findings"] == 0
    assert "score" not in by["GPT"]


def test_reproduction_title_follows_the_data():
    single = [_f(f"S{i}", ["GPT"]) for i in range(7)] + [_f("M1", ["GPT", "Claude"])]
    rep = reproduction(single)
    assert rep["title"].startswith("Most high-significance disclosures are model-specific")
    assert rep["single"] == 7 and rep["multi"] == 1
    assert 2 <= len(rep["notes"]) <= 3
    assert [n["marker"] for n in rep["notes"]] == list(range(1, len(rep["notes"]) + 1))

    shared = [_f(f"M{i}", ["GPT", "Claude"]) for i in range(7)] + [_f("S1", ["GPT"])]
    assert reproduction(shared)["title"] == (
        "Most high-significance disclosures are reproduced by more than one model."
    )


def test_overview_and_plan_are_computed_not_fixed():
    findings = [
        _f("C1", ["GPT", "Claude"], url=True),
        _f("C2", ["GPT"], sens=2),
        _f("C3", ["Claude"], status="CONTESTED"),
    ]
    ov = exposure_overview(findings)
    assert ov["totals"]["findings"] == 3
    assert ov["totals"]["corroborated_pct"] == 33
    plan = verification_plan(findings)
    counts = {item["title"]: item["count"] for item in plan["items"]}
    assert counts["Confirm the sourced high-significance findings"] == 1
    assert counts["Resolve contested findings"] == 1
    assert plan["items"][-1]["exceptional"] is True
