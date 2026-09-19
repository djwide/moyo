"""Post-run checks before auto-delivering a raw Exposure Data report.

A run passes when all of the following hold:

- more than 75% of the models requested for the run were tested (ok or partial)
- there were no fatal errors
- required report sections are present
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

MIN_COVERAGE = 0.75  # must be strictly greater than this
TESTED_STATUSES = frozenset({"ok", "partial"})
REQUIRED_REPORT_DATA_KEYS = ("topic", "findings", "counts")


@dataclass
class ValidationResult:
    ok: bool
    models_requested: int = 0
    models_tested: int = 0
    coverage: float = 0.0
    missing_sections: list[str] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    models_failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            failed = (
                f"; {len(self.models_failed)} model(s) failed retrieval"
                if self.models_failed
                else ""
            )
            return (
                f"validation passed: {self.models_tested}/{self.models_requested} "
                f"models tested ({self.coverage:.0%}){failed}"
            )
        bits = list(self.reasons) or ["validation failed"]
        return "; ".join(bits)[:2000]

    def to_firestore(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "modelsRequested": self.models_requested,
            "modelsTested": self.models_tested,
            "coverage": self.coverage,
            "missingSections": list(self.missing_sections),
            "fatalErrors": list(self.fatal_errors),
            "modelsFailed": list(self.models_failed),
            "reasons": list(self.reasons),
            "summary": self.summary(),
        }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_path(artifacts: Mapping[str, Path], name: str) -> Path | None:
    path = artifacts.get(name)
    if path is not None and path.is_file():
        return path
    return None


def _count_jsonl_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _is_fatal_issue(item: Mapping[str, Any]) -> bool:
    if item.get("fatal") is True:
        return True
    severity = str(item.get("severity") or item.get("type") or "").strip().lower()
    if severity == "fatal":
        return True
    reason = str(item.get("reason") or "").lower()
    return " fatal" in f" {reason}" or reason.startswith("fatal")


def _fatal_from_issues(items: Iterable[Any], *, label: str) -> list[str]:
    out: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        if not _is_fatal_issue(item):
            continue
        reason = str(item.get("reason") or "fatal error").strip() or "fatal error"
        source = str(item.get("source") or item.get("source_model") or "").strip()
        prefix = f"{label} {source}: " if source else f"{label}: "
        out.append(f"{prefix}{reason}"[:240])
    return out


def _model_stats(check_path: Path | None) -> tuple[int, int, list[str], list[str]]:
    """Return (requested, tested, failed_labels, fatals)."""
    if check_path is None:
        return 0, 0, [], ["missing llm-retrieval-check.json"]
    try:
        payload = _read_json(check_path)
    except (OSError, json.JSONDecodeError) as exc:
        return 0, 0, [], [f"unreadable llm-retrieval-check.json: {exc}"]
    if not isinstance(payload, dict):
        return 0, 0, [], ["llm-retrieval-check.json is not an object"]

    retrieval = payload.get("retrieval")
    if not isinstance(retrieval, list):
        retrieval = []
    requested = payload.get("retrieval_total")
    try:
        requested_n = int(requested) if requested is not None else len(retrieval)
    except (TypeError, ValueError):
        requested_n = len(retrieval)
    if requested_n <= 0:
        requested_n = len(retrieval)

    tested = 0
    failed: list[str] = []
    for row in retrieval:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").strip().lower()
        name = str(row.get("name") or "unknown").strip() or "unknown"
        if status in TESTED_STATUSES:
            tested += 1
        elif status == "fail":
            failed.append(name)
    return requested_n, tested, failed, []


def _section_problems(artifacts: Mapping[str, Path]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    fatals: list[str] = []

    check = _artifact_path(artifacts, "llm-retrieval-check.json")
    claims = _artifact_path(artifacts, "claims.jsonl")
    report_data_path = _artifact_path(artifacts, "report_data.json")
    exploration = _artifact_path(artifacts, "exploration.md")

    if check is None:
        missing.append("llm-retrieval-check.json")
        fatals.append("missing llm-retrieval-check.json")
    if exploration is None:
        missing.append("exploration.md")
        fatals.append("missing exploration.md")
    if claims is None:
        missing.append("claims.jsonl")
    elif _count_jsonl_rows(claims) <= 0:
        missing.append("claims.jsonl (empty)")

    if report_data_path is None:
        missing.append("report_data.json")
    else:
        try:
            data = _read_json(report_data_path)
        except (OSError, json.JSONDecodeError) as exc:
            fatals.append(f"unreadable report_data.json: {exc}")
            missing.append("report_data.json")
            data = None
        if not isinstance(data, dict):
            if data is not None:
                missing.append("report_data.json")
                fatals.append("report_data.json is not an object")
        else:
            for key in REQUIRED_REPORT_DATA_KEYS:
                if key not in data:
                    missing.append(f"report_data.{key}")
            topic = str(data.get("topic") or "").strip() if isinstance(data, dict) else ""
            if "topic" in data and not topic:
                missing.append("report_data.topic (empty)")
            if "findings" in data and not isinstance(data.get("findings"), list):
                missing.append("report_data.findings")
            if "counts" in data and not isinstance(data.get("counts"), dict):
                missing.append("report_data.counts")
            fatals.extend(
                _fatal_from_issues(data.get("collection_issues") or [], label="collection")
            )
            meta = data.get("explore_meta") or {}
            if isinstance(meta, dict):
                fatals.extend(
                    _fatal_from_issues(meta.get("collection_issues") or [], label="collection")
                )

    extract_path = _artifact_path(artifacts, "extract_issues.json")
    if extract_path is not None:
        try:
            extra = _read_json(extract_path)
        except (OSError, json.JSONDecodeError) as exc:
            fatals.append(f"unreadable extract_issues.json: {exc}")
            extra = []
        if isinstance(extra, list):
            fatals.extend(_fatal_from_issues(extra, label="extract"))

    return missing, fatals


def validate_artifacts(artifacts: Mapping[str, Path]) -> ValidationResult:
    """Validate one prompt's collected artifacts."""
    missing, fatals = _section_problems(artifacts)
    requested, tested, failed, check_fatals = _model_stats(
        _artifact_path(artifacts, "llm-retrieval-check.json")
    )
    fatals = list(dict.fromkeys([*check_fatals, *fatals]))

    coverage = (tested / requested) if requested > 0 else 0.0
    reasons: list[str] = []
    if requested <= 0:
        reasons.append("no retrieval models were requested")
    elif coverage <= MIN_COVERAGE:
        reasons.append(
            f"tested {tested}/{requested} models ({coverage:.0%}); "
            f"need more than {MIN_COVERAGE:.0%}"
        )
    if fatals:
        reasons.extend(fatals)
    if missing:
        reasons.append("missing sections: " + ", ".join(missing))

    ok = (
        requested > 0
        and coverage > MIN_COVERAGE
        and not fatals
        and not missing
    )
    return ValidationResult(
        ok=ok,
        models_requested=requested,
        models_tested=tested,
        coverage=coverage,
        missing_sections=missing,
        fatal_errors=fatals,
        reasons=reasons,
        models_failed=failed,
    )


def validate_prompt_runs(runs: Iterable[Any]) -> ValidationResult:
    """AND-merge per-prompt results. One failing prompt fails the order."""
    results = [validate_artifacts(getattr(run, "artifacts", {}) or {}) for run in runs]
    if not results:
        return ValidationResult(
            ok=False,
            reasons=["no prompt runs to validate"],
            fatal_errors=["no prompt runs to validate"],
        )
    if len(results) == 1:
        return results[0]

    requested = max(item.models_requested for item in results)
    tested = min(item.models_tested for item in results)
    coverage = min(item.coverage for item in results)
    missing = list(dict.fromkeys(name for item in results for name in item.missing_sections))
    fatals = list(dict.fromkeys(name for item in results for name in item.fatal_errors))
    reasons = list(dict.fromkeys(name for item in results for name in item.reasons))
    failed = list(dict.fromkeys(name for item in results for name in item.models_failed))
    ok = all(item.ok for item in results)
    if not ok and not reasons:
        reasons = ["one or more prompts failed validation"]
    return ValidationResult(
        ok=ok,
        models_requested=requested,
        models_tested=tested,
        coverage=coverage,
        missing_sections=missing,
        fatal_errors=fatals,
        reasons=reasons,
        models_failed=failed,
    )
