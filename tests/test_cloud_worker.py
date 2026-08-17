"""Unit tests for cloud_worker helpers (no Firebase / network)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import cloud_worker as cw


def test_storage_bucket_name_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("STORAGE_BUCKET", "my-bucket")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "senteguard-website")
    assert cw._storage_bucket_name() == "my-bucket"


def test_storage_bucket_name_defaults_to_firebase_app(monkeypatch):
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    monkeypatch.delenv("FIREBASE_STORAGE_BUCKET", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "senteguard-website")
    assert cw._storage_bucket_name() == "senteguard-website.firebasestorage.app"


def test_normalize_product_aliases():
    assert cw.normalize_product(None) == "snapshot"
    assert cw.normalize_product("exposure") == "snapshot"
    assert cw.normalize_product("basis") == "basis"
    assert cw.normalize_product("Basis Report") == "basis"
    assert cw.normalize_product("all") == "both"
    with pytest.raises(ValueError):
        cw.normalize_product("deluxe")


def test_orders_collection_defaults_to_reports(monkeypatch):
    monkeypatch.delenv("FIRESTORE_ORDERS_COLLECTION", raising=False)
    monkeypatch.delenv("FIRESTORE_COLLECTION", raising=False)
    assert cw._orders_collection_candidates()[0] == "reports"


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


def test_parse_live_reports_collection_shape():
    spec = cw.parse_order(
        "ord_781a0fe4e2d8a38c048823154ff0ec16",
        {
            "orderId": "ord_781a0fe4e2d8a38c048823154ff0ec16",
            "product": "snapshot",
            "productId": "moyo_snapshot",
            "prompts": ["What controversies happened related to"],
            "customerPrompts": ["What controversies happened related to"],
            "paymentStatus": "paid",
            "reportStatus": "queued",
            "qcStatus": "pending",
        },
    )
    assert spec.product == "snapshot"
    assert spec.prompts == ["What controversies happened related to"]
    assert spec.payment_status == "paid"
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
    (tmp_path / "llm-retrieval-check.md").write_text("# LLM Retrieval Check\n", encoding="utf-8")
    (tmp_path / "llm-retrieval-check.json").write_text("{}", encoding="utf-8")

    found = cw.collect_artifacts(tmp_path, run_dir, "snapshot")
    assert set(cw.CONTRACT_ARTIFACTS) <= set(found)
    assert found["llm-retrieval-check.md"].name == "llm-retrieval-check.md"
    assert found["llm-retrieval-check.json"].name == "llm-retrieval-check.json"
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


def test_assert_explore_produced_content_fails_when_empty(tmp_path: Path):
    (tmp_path / "raw_responses.json").write_text(
        json.dumps(
            [
                {"source_label": "GPT", "error": "401 Unauthorized", "text": ""},
                {"source_label": "Claude", "error": "missing api key", "text": ""},
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "exploration.md").write_text(
        "# Topic\n\n> Retrieval failed: 401\n> Retrieval failed: key\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="0 usable LLM answers"):
        cw.assert_explore_produced_content(tmp_path, "Enron?")


def test_assert_explore_produced_content_ok(tmp_path: Path):
    (tmp_path / "raw_responses.json").write_text(
        json.dumps([{"source_label": "GPT", "text": "Enron hid debt via SPEs."}]),
        encoding="utf-8",
    )
    cw.assert_explore_produced_content(tmp_path, "Enron?")


def test_assert_report_has_claims_fails_when_empty(tmp_path: Path):
    (tmp_path / "claims.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "chunks.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="0 claims"):
        cw.assert_report_has_claims(tmp_path, "Enron?")


def test_required_llm_env_presence(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    presence = cw._required_llm_env_presence()
    assert presence["OPENAI_API_KEY"] is True
    assert presence["MOONSHOT_API_KEY"] is False


def test_retrieval_check_storage_paths_single_and_multi(tmp_path: Path):
    slug = tmp_path / "01_enron"
    slug.mkdir()
    md = slug / "llm-retrieval-check.md"
    js = slug / "llm-retrieval-check.json"
    md.write_text("# check\n", encoding="utf-8")
    js.write_text("{}\n", encoding="utf-8")
    paths = dict(cw.retrieval_check_storage_paths("ord_x", tmp_path))
    assert paths["reports/ord_x/llm-retrieval-check.md"] == md
    assert paths["reports/ord_x/01_enron/llm-retrieval-check.md"] == md
    assert "reports/ord_x/llm-retrieval-check.json" in paths

    other = tmp_path / "02_other"
    other.mkdir()
    (other / "llm-retrieval-check.md").write_text("b", encoding="utf-8")
    paths2 = [p for p, _ in cw.retrieval_check_storage_paths("ord_x", tmp_path)]
    assert "reports/ord_x/llm-retrieval-check.md" not in paths2
    assert "reports/ord_x/01_enron/llm-retrieval-check.md" in paths2
    assert "reports/ord_x/02_other/llm-retrieval-check.md" in paths2


def test_write_report_config_overlays_hosted_cluster(tmp_path: Path, monkeypatch):
    from moyo.llm.testing import enable_test_mode

    enable_test_mode(False)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.delenv("MOYO_TEST_MODE", raising=False)
    spec = cw.OrderSpec(order_id="ord_x", prompts=["Who killed JFK?"], headline="Cover")
    dest = cw._write_report_config(tmp_path, spec, "ord_x__01")
    import yaml

    cfg = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert cfg["extract"]["provider"] == "openai"
    assert cfg["extract"]["model"] == "gpt-4o"
    assert cfg["extract"]["api_key"] == "$OPENAI_API_KEY"
    assert cfg["extract"].get("prompt") == "prompts/extract_claims.md"
    assert cfg["synthesize"]["provider"] == "openai"
    assert cfg["synthesize"]["api_key"] == "$OPENAI_API_KEY"
    assert cfg["cluster"]["provider"] == "openai"
    assert cfg["cluster"]["api_key"] == "$OPENAI_API_KEY"
    assert "sk-test-openai" not in dest.read_text(encoding="utf-8")
    assert cfg["render"]["headline"] == "Cover"
