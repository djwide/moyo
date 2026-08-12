"""Clustering should raise confidence for multi-LLM and multi-source agreement."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports"))

from pipeline.cluster import cluster_claims, confidence_boost  # noqa: E402


def _claim(
    claim_id: str,
    *,
    text: str,
    model: str,
    confidence: int = 3,
    citations: list | None = None,
) -> dict:
    return {
        "claim_id": claim_id,
        "claim": text,
        "source_model": model,
        "query_id": "Q01",
        "confidence": confidence,
        "corroboration": 1,
        "interestingness": 3,
        "specificity": 3,
        "sensitivity": 3,
        "status": "UNVERIFIED",
        "citations": citations or [],
    }


def test_confidence_boost_formula() -> None:
    assert confidence_boost(n_models=1, n_citations=0) == 0
    assert confidence_boost(n_models=2, n_citations=0) == 1
    assert confidence_boost(n_models=3, n_citations=0) == 2
    assert confidence_boost(n_models=1, n_citations=2) == 1
    assert confidence_boost(n_models=2, n_citations=3) == 3
    assert confidence_boost(n_models=3, n_citations=3) == 4


def test_single_model_no_citations_keeps_confidence() -> None:
    claims = [
        _claim("C1", text="Congressman accepted donations from oil PACs", model="GPT", confidence=3),
    ]
    out, clusters = cluster_claims(claims)
    assert out[0]["confidence"] == 3
    assert out[0]["corroboration"] == 1
    assert out[0]["source_count"] == 0
    assert clusters[0]["size"] == 1


def test_multi_llm_raises_confidence_and_corroboration() -> None:
    text = "Congressman accepted donations from oil PACs in 2024"
    claims = [
        _claim("C1", text=text, model="GPT", confidence=3),
        _claim("C2", text=text, model="Gemini", confidence=3),
        _claim("C3", text=text, model="Grok", confidence=3),
    ]
    out, _ = cluster_claims(claims)
    assert all(c["corroboration"] == 3 for c in out)
    assert all(c["status"] == "CORROBORATED" for c in out)
    # base 3 +2 (3 LLMs) = 5
    assert all(c["confidence"] == 5 for c in out)


def test_multi_citation_raises_confidence() -> None:
    text = "Congressman accepted donations from oil PACs in 2024"
    claims = [
        _claim(
            "C1",
            text=text,
            model="GPT",
            confidence=2,
            citations=["https://opensecrets.org/a", "https://fec.gov/b"],
        ),
    ]
    out, clusters = cluster_claims(claims)
    assert out[0]["source_count"] == 2
    assert out[0]["corroboration"] == 1
    # base 2 +1 (≥2 citations) = 3
    assert out[0]["confidence"] == 3
    assert clusters[0]["citations"] == [
        "https://fec.gov/b",
        "https://opensecrets.org/a",
    ]


def test_multi_llm_and_citations_stack_capped_at_five() -> None:
    text = "Congressman accepted donations from oil PACs in 2024"
    claims = [
        _claim(
            "C1",
            text=text,
            model="GPT",
            confidence=3,
            citations=["https://opensecrets.org/a"],
        ),
        _claim(
            "C2",
            text=text,
            model="Gemini",
            confidence=3,
            citations=["https://fec.gov/b", {"url": "https://propublica.org/c"}],
        ),
        _claim(
            "C3",
            text=text,
            model="Grok",
            confidence=3,
            citations=["https://opensecrets.org/a"],
        ),
    ]
    out, _ = cluster_claims(claims)
    assert out[0]["corroboration"] == 3
    assert out[0]["source_count"] == 3
    # base 3 +2 (models) +2 (citations) = 7 → capped at 5
    assert all(c["confidence"] == 5 for c in out)
