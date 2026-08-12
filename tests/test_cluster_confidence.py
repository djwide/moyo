"""Ollama-backed claim collapse: similar meaning → one claim."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports"))

from pipeline.cluster import (  # noqa: E402
    cluster_claims,
    confidence_from_models,
    has_exact_number,
    merge_claim_group,
    specificity_with_numbers,
    _parse_groups_payload,
)


def _claim(
    claim_id: str,
    *,
    text: str,
    model: str,
    confidence: int = 3,
    citations: list | None = None,
    sensitivity: int = 3,
    specificity: int = 3,
    novelty: int = 3,
    interestingness: int = 3,
) -> dict:
    return {
        "claim_id": claim_id,
        "claim": text,
        "source_model": model,
        "query_id": "Q01",
        "confidence": confidence,
        "corroboration": 1,
        "interestingness": interestingness,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "novelty": novelty,
        "status": "UNVERIFIED",
        "citations": citations or [],
        "raw_excerpt": text[:80],
    }


def test_confidence_from_models() -> None:
    assert confidence_from_models(1) == 1
    assert confidence_from_models(3) == 3
    assert confidence_from_models(9) == 5


def test_exact_number_boosts_specificity() -> None:
    assert has_exact_number("donated $100,001 to PACs")
    assert specificity_with_numbers(3, "donated $250,000") == 4


def test_parse_groups_payload() -> None:
    valid = {"C1", "C2", "C3"}
    raw = json.dumps({"groups": [["C1", "C2"], ["C3"]]})
    assert _parse_groups_payload(raw, valid) == [["C1", "C2"], ["C3"]]
    # Missing C3 becomes singleton
    raw2 = json.dumps({"groups": [["C1", "C2"]]})
    groups = _parse_groups_payload(raw2, valid)
    assert ["C1", "C2"] in groups
    assert ["C3"] in groups


def test_dry_run_groups_identical_text() -> None:
    text = "Congressman accepted donations from oil PACs in 2024"
    claims = [
        _claim("C1", text=text, model="GPT", sensitivity=3),
        _claim("C2", text=text, model="Gemini", sensitivity=4),
        _claim("C3", text="Unrelated Coca-Cola formula claim", model="Grok"),
    ]
    out, clusters = cluster_claims(claims, dry_run=True)
    assert len(out) == 2
    multi = next(c for c in out if c["merged_count"] == 2)
    assert multi["sensitivity"] == 4
    assert multi["confidence"] == 2
    assert {m["sensitivity"] for m in multi["member_scores"]} == {3, 4}
    assert len(clusters) == 2


def test_injected_group_fn_merges() -> None:
    text = "Gonzalez held a Bank of China account worth $100,001"
    claims = [
        _claim("C1", text=text, model="Gemini", citations=["Texas Tribune"], specificity=3),
        _claim(
            "C2",
            text="Gonzalez held Bank of China account ($100k–$250k)",
            model="Claude",
            citations=["House Clerk"],
            specificity=4,
            interestingness=5,
        ),
        _claim("C3", text="Late STOCK Act filings", model="GPT"),
    ]

    def group_fn(rows: list[dict]) -> list[list[str]]:
        return [["C1", "C2"], ["C3"]]

    out, _ = cluster_claims(claims, group_fn=group_fn)
    assert len(out) == 2
    bank = next(c for c in out if c["merged_count"] == 2)
    assert bank["confidence"] == 2
    assert bank["source_count"] == 2
    assert bank["sensitivity"] == 3  # both default 3; max is 3
    assert len(bank["member_scores"]) == 2


def test_dedupe_findings_by_group_keeps_one_per_cluster() -> None:
    from pipeline.cluster import dedupe_findings_by_group

    findings = [
        {
            "claim_id": "C1",
            "cluster_id": "CL001",
            "claim": "short",
            "merged_count": 1,
            "corroboration": 1,
            "sensitivity": 2,
            "specificity": 2,
            "interestingness": 2,
            "merged_from": ["C1"],
        },
        {
            "claim_id": "C2",
            "cluster_id": "CL001",
            "claim": "longer survivor text",
            "merged_count": 3,
            "corroboration": 3,
            "sensitivity": 4,
            "specificity": 4,
            "interestingness": 4,
            "merged_from": ["C1", "C2", "C9"],
        },
        {
            "claim_id": "C9",
            "cluster_id": "CL001",
            "claim": "raw member that should hide",
            "merged_count": 1,
            "corroboration": 1,
            "sensitivity": 1,
            "specificity": 1,
            "interestingness": 1,
            "merged_from": ["C9"],
        },
        {
            "claim_id": "C3",
            "cluster_id": "CL002",
            "claim": "other group",
            "merged_count": 1,
            "corroboration": 1,
            "sensitivity": 3,
            "specificity": 3,
            "interestingness": 3,
            "merged_from": ["C3"],
        },
    ]
    out = dedupe_findings_by_group(findings)
    assert [f["claim_id"] for f in out] == ["C2", "C3"]


def test_merge_claim_group_direct() -> None:
    members = [
        _claim("C1", text="Paid $2,400 per month for office space", model="Qwen", sensitivity=2),
        _claim("C2", text="Paid $2,400 per month for office space", model="Grok", sensitivity=5),
    ]
    merged = merge_claim_group(members, cluster_id="CL001")
    assert merged["sensitivity"] == 5
    assert merged["confidence"] == 2
    assert len(merged["member_scores"]) == 2
