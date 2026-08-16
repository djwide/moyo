"""Unit tests for cloud_worker helpers (no Firebase / network)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import cloud_worker as cw


def test_normalize_product_aliases():
    assert cw.normalize_product(None) == "snapshot"
    assert cw.normalize_product("exposure") == "snapshot"
    assert cw.normalize_product("basis") == "basis"
    assert cw.normalize_product("Basis Report") == "basis"
    assert cw.normalize_product("all") == "both"
    with pytest.raises(ValueError):
        cw.normalize_product("deluxe")


def test_parse_storefront_order():
    spec = cw.parse_order(
        "ord_1",
        {
            "createdAt": "2026-08-16T00:00:00Z",
            "currency": "usd",
            "customerEmail": "customer@example.com",
            "generationFinishedAt": None,
            "generationStartedAt": None,
            "paidAt": "2026-08-16T00:01:00Z",
            "paymentStatus": "paid",
            "product": "basis",
            "prompts": ["Who killed JFK?", "What is the recipe for Coca-Cola?"],
            "qcStatus": "pending",
            "reportStatus": "awaiting_prompts",
            "stripeSessionID": "cs_live_x",
        },
    )
    assert spec.product == "basis"
    assert spec.prompts == [
        "Who killed JFK?",
        "What is the recipe for Coca-Cola?",
    ]
    assert spec.customer_email == "customer@example.com"
    assert spec.payment_status == "paid"


def test_parse_order_prompts_json_string():
    spec = cw.parse_order(
        "ord_2",
        {"product": "basis", "prompts": '["Alpha secret", "Beta secret"]'},
    )
    assert spec.prompts == ["Alpha secret", "Beta secret"]


def test_parse_order_empty_prompts_json():
    with pytest.raises(ValueError, match="awaiting_prompts"):
        cw.parse_order("x", {"product": "basis", "prompts": "[]"})


def test_parse_order_requires_prompts():
    with pytest.raises(ValueError, match="awaiting_prompts"):
        cw.parse_order("x", {"product": "snapshot"})


def test_prompt_slug_is_stable():
    assert cw.prompt_slug(1, "Who killed JFK?").startswith("01_who_killed_jfk")
    assert cw.prompt_slug(2, "Who killed JFK?").startswith("02_")


def test_serialize_raw_responses():
    @dataclass
    class Row:
        seed: str
        text: str
        llm_label: str = "GPT"

    @dataclass
    class Result:
        prompt: str
        results: list

    rows = cw.serialize_raw_responses(
        [Result(prompt="q1", results=[Row(seed="s", text="hello")])]
    )
    assert rows[0]["prompt"] == "q1"
    assert rows[0]["text"] == "hello"


def test_collect_artifacts_and_evidence(tmp_path: Path):
    run_dir = tmp_path / "report_runs" / "ord"
    output = run_dir / "output"
    output.mkdir(parents=True)
    (run_dir / "report.md").write_text("# md\n", encoding="utf-8")
    (output / "report.html").write_text("<html/>", encoding="utf-8")
    (output / "report.pdf").write_bytes(b"%PDF")
    (run_dir / "claims.jsonl").write_text(
        json.dumps({"claim_id": "C1", "claim": "fact"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "report_data.json").write_text(
        json.dumps({"topic": "t", "headline": "H", "findings": [{"id": 1}]}),
        encoding="utf-8",
    )
    (tmp_path / "raw_responses.json").write_text("[]", encoding="utf-8")
    evidence = cw.build_evidence(run_dir, prompt="Who killed JFK?")
    (tmp_path / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")

    found = cw.collect_artifacts(tmp_path, run_dir, "snapshot")
    assert set(cw.CONTRACT_ARTIFACTS) <= set(found)
    assert evidence["prompt"] == "Who killed JFK?"
    assert evidence["headline"] == "H"
    assert evidence["claims"][0]["claim_id"] == "C1"


def test_collect_artifacts_basis_falls_back(tmp_path: Path):
    run_dir = tmp_path / "run"
    output = run_dir / "output"
    output.mkdir(parents=True)
    (run_dir / "report.md").write_text("md", encoding="utf-8")
    (output / "basis-report.pdf").write_bytes(b"%PDF")
    (output / "basis-report.html").write_text("<html/>", encoding="utf-8")
    (tmp_path / "raw_responses.json").write_text("[]", encoding="utf-8")
    (tmp_path / "evidence.json").write_text("{}", encoding="utf-8")
    found = cw.collect_artifacts(tmp_path, run_dir, "basis")
    assert found["report.pdf"].name == "basis-report.pdf"
    assert found["report.html"].name == "basis-report.html"


def test_storage_destinations_single_prompt_also_writes_flat_qc_path(tmp_path: Path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF")
    run = cw.PromptRun(
        index=1,
        prompt="Who killed JFK?",
        slug="01_who_killed_jfk",
        run_id="ord__01_who_killed_jfk",
        artifacts={"report.pdf": pdf},
    )
    dest = dict(cw.storage_destinations("ord", [run]))
    assert dest["reports/ord/report.pdf"] == pdf
    assert dest["reports/ord/01_who_killed_jfk/report.pdf"] == pdf


def test_storage_destinations_multi_prompt_no_flat_collision(tmp_path: Path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    runs = [
        cw.PromptRun(1, "A", "01_a", "ord__01_a", {"report.pdf": a}),
        cw.PromptRun(2, "B", "02_b", "ord__02_b", {"report.pdf": b}),
    ]
    paths = [p for p, _ in cw.storage_destinations("ord", runs)]
    assert "reports/ord/report.pdf" not in paths
    assert "reports/ord/01_a/report.pdf" in paths
    assert "reports/ord/02_b/report.pdf" in paths
    manifest = cw.artifact_manifest("ord", runs)
    assert len(manifest["reports"]) == 2
    assert manifest["reports"][0]["prefix"] == "reports/ord/01_a/"
