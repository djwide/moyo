from __future__ import annotations

import json
from pathlib import Path

from report_validation import ValidationResult, validate_artifacts, validate_prompt_runs


def _write_run(
    tmp_path: Path,
    *,
    retrieval: list[dict],
    claims: int = 2,
    topic: str = "Enron",
    findings: list | None = None,
    counts: dict | None = None,
    extra_files: dict[str, str] | None = None,
    omit: set[str] | None = None,
    collection_issues: list | None = None,
) -> dict[str, Path]:
    omit = omit or set()
    artifacts: dict[str, Path] = {}
    if "llm-retrieval-check.json" not in omit:
        payload = {
            "prompt": topic,
            "retrieval": retrieval,
            "retrieval_ok": sum(1 for row in retrieval if row.get("status") == "ok"),
            "retrieval_total": len(retrieval),
        }
        path = tmp_path / "llm-retrieval-check.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        artifacts["llm-retrieval-check.json"] = path
    if "claims.jsonl" not in omit:
        path = tmp_path / "claims.jsonl"
        lines = [json.dumps({"claim_id": f"C{i}", "claim": f"fact {i}"}) for i in range(claims)]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        artifacts["claims.jsonl"] = path
    if "report_data.json" not in omit:
        data = {
            "topic": topic,
            "findings": findings if findings is not None else [{"claim_id": "C0"}],
            "counts": counts if counts is not None else {"findings": max(claims, 1)},
        }
        if collection_issues is not None:
            data["collection_issues"] = collection_issues
        path = tmp_path / "report_data.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        artifacts["report_data.json"] = path
    if "exploration.md" not in omit:
        path = tmp_path / "exploration.md"
        path.write_text("# exploration\n", encoding="utf-8")
        artifacts["exploration.md"] = path
    for name, text in (extra_files or {}).items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        artifacts[name] = path
    return artifacts


def _ten_models(*, fail: int = 0, partial: int = 0) -> list[dict]:
    rows = []
    for i in range(10):
        if i < fail:
            status = "fail"
        elif i < fail + partial:
            status = "partial"
        else:
            status = "ok"
        rows.append({"name": f"model-{i}", "status": status, "reason": ""})
    return rows


def test_all_models_ok_passes(tmp_path: Path):
    result = validate_artifacts(_write_run(tmp_path, retrieval=_ten_models()))
    assert result.ok
    assert result.models_tested == 10
    assert result.models_requested == 10
    assert result.fatal_errors == []
    assert result.missing_sections == []


def test_one_model_fail_still_passes(tmp_path: Path):
    result = validate_artifacts(_write_run(tmp_path, retrieval=_ten_models(fail=1)))
    assert result.ok
    assert result.models_tested == 9
    assert result.coverage == 0.9
    assert result.models_failed == ["model-0"]


def test_partial_counts_as_tested(tmp_path: Path):
    result = validate_artifacts(
        _write_run(tmp_path, retrieval=_ten_models(fail=1, partial=1))
    )
    assert result.ok
    assert result.models_tested == 9


def test_enough_failures_invalidates(tmp_path: Path):
    result = validate_artifacts(_write_run(tmp_path, retrieval=_ten_models(fail=4)))
    assert not result.ok
    assert result.models_tested == 6
    assert any("75%" in reason for reason in result.reasons)


def test_missing_required_sections_fail(tmp_path: Path):
    result = validate_artifacts(
        _write_run(tmp_path, retrieval=_ten_models(), omit={"claims.jsonl", "report_data.json"})
    )
    assert not result.ok
    assert "claims.jsonl" in result.missing_sections
    assert "report_data.json" in result.missing_sections


def test_empty_claims_is_missing_section(tmp_path: Path):
    result = validate_artifacts(
        _write_run(tmp_path, retrieval=_ten_models(), claims=0)
    )
    assert not result.ok
    assert any("claims.jsonl" in item for item in result.missing_sections)


def test_fatal_collection_issue_fails(tmp_path: Path):
    result = validate_artifacts(
        _write_run(
            tmp_path,
            retrieval=_ten_models(),
            collection_issues=[{"fatal": True, "reason": "pipeline aborted"}],
        )
    )
    assert not result.ok
    assert result.fatal_errors
    assert any("pipeline aborted" in err for err in result.fatal_errors)


def test_missing_retrieval_check_is_fatal(tmp_path: Path):
    result = validate_artifacts(
        _write_run(tmp_path, retrieval=_ten_models(), omit={"llm-retrieval-check.json"})
    )
    assert not result.ok
    assert any("llm-retrieval-check.json" in err for err in result.fatal_errors)


def test_validate_prompt_runs_fails_if_any_prompt_fails(tmp_path: Path):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad"
    good_dir.mkdir()
    bad_dir.mkdir()

    class Run:
        def __init__(self, artifacts):
            self.artifacts = artifacts

    good = Run(_write_run(good_dir, retrieval=_ten_models()))
    bad = Run(_write_run(bad_dir, retrieval=_ten_models(fail=5)))
    merged = validate_prompt_runs([good, bad])
    assert not merged.ok
    assert merged.models_tested == 5


def test_validation_result_firestore_shape():
    payload = ValidationResult(
        ok=True,
        models_requested=10,
        models_tested=9,
        coverage=0.9,
        models_failed=["Grok"],
    ).to_firestore()
    assert payload["ok"] is True
    assert payload["modelsRequested"] == 10
    assert payload["modelsTested"] == 9
    assert "Grok" in payload["modelsFailed"]
