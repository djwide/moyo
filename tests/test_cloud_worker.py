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


def test_storage_bucket_name_prefers_moyo_reports_env(monkeypatch):
    monkeypatch.setenv("MOYO_REPORTS_STORAGE_BUCKET", "gs://senteguard-website-moyo-reports/")
    monkeypatch.setenv("STORAGE_BUCKET", "other-bucket")
    assert cw._storage_bucket_name() == "senteguard-website-moyo-reports"


def test_storage_bucket_name_defaults_to_dedicated_reports_bucket(monkeypatch):
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    monkeypatch.delenv("MOYO_REPORTS_STORAGE_BUCKET", raising=False)
    monkeypatch.delenv("FIREBASE_STORAGE_BUCKET", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "senteguard-website")
    assert cw._storage_bucket_name() == "senteguard-website-moyo-reports"


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
    assert spec.qc_required is True
    assert spec.product_id == "moyo_basis"
    assert spec.generation_mode == "full"


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
    assert spec.qc_required is True
    spec = cw.parse_order(
        "ord_2",
        {"product": "basis", "prompts": '["Alpha secret", "Beta secret"]'},
    )
    assert spec.prompts == ["Alpha secret", "Beta secret"]


def test_normalize_generation_mode():
    assert cw.normalize_generation_mode(None) == "full"
    assert cw.normalize_generation_mode("pdf_from_markdown") == "pdf_from_markdown"
    assert cw.normalize_generation_mode("rebuild_graphics") == "rebuild_graphics"
    assert cw.normalize_generation_mode("graphics-only") == "rebuild_graphics"


def test_parse_order_generation_mode():
    spec = cw.parse_order(
        "ord_1",
        {
            "product": "snapshot",
            "prompts": ["Enron"],
            "generationMode": "pdf_from_markdown",
        },
    )
    assert spec.generation_mode == "pdf_from_markdown"


def test_parse_order_agent_source_skips_qc_when_field_missing():
    spec = cw.parse_order(
        "ord_x402",
        {
            "product": "snapshot",
            "prompts": ["Enron"],
            "source": "x402",
        },
    )
    assert spec.qc_required is False
    spec = cw.parse_order(
        "ord_agent",
        {
            "product": "snapshot",
            "productId": "moyo_snapshot",
            "prompts": ["Enron"],
            "qcRequired": False,
            "source": "x402",
        },
    )
    assert spec.qc_required is False
    assert spec.product_id == "moyo_snapshot"
    assert spec.source == "x402"


def test_normalize_qc_required_falls_back_to_source():
    assert cw.normalize_qc_required(None, "stripe_checkout") is True
    assert cw.normalize_qc_required(None, "admin") is True
    assert cw.normalize_qc_required(None, "x402") is False
    assert cw.normalize_qc_required(None, "stripe_mpp") is False
    assert cw.normalize_qc_required(False, "stripe_checkout") is False
    assert cw.normalize_qc_required("true", "x402") is True


def test_is_awaiting_qc_status_accepts_legacy_qc_pending():
    assert cw.is_awaiting_qc_status("awaiting_qc") is True
    assert cw.is_awaiting_qc_status("qc_pending") is True
    assert cw.is_awaiting_qc_status("QC-pending") is True
    assert cw.is_awaiting_qc_status("delivered") is False
    assert cw.is_awaiting_qc_status("generating") is False


def test_rebuild_topic_dirs_single(tmp_path: Path):
    (tmp_path / "report.md").write_text("# hi\n", encoding="utf-8")
    spec = cw.parse_order("ord_1", {"product": "snapshot", "prompts": ["Enron"]})
    topics = cw.rebuild_topic_dirs(tmp_path, spec)
    assert len(topics) == 1
    assert topics[0][2] == tmp_path


def test_copy_rebuild_sources_keeps_yaml_and_assets(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "report.yaml").write_text("title: edited\n", encoding="utf-8")
    (src / "report.md").write_text("# edited\n", encoding="utf-8")
    (src / "assets").mkdir()
    (src / "assets" / "exposure-radar.svg").write_text("<svg />", encoding="utf-8")
    run_dir = tmp_path / "run"
    prompt_dir = tmp_path / "prompt"
    cw.copy_rebuild_sources(src, run_dir, prompt_dir)
    assert (run_dir / "report.yaml").read_text(encoding="utf-8") == "title: edited\n"
    assert (run_dir / "assets" / "exposure-radar.svg").is_file()


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
        json.dumps(
            {
                "topic": "t",
                "headline": "H",
                "findings": [
                    {
                        "claim_id": "C1",
                        "claim": "fact",
                        "citations": ["https://example.com/source"],
                        "source_models": ["GPT"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "raw_responses.json").write_text("[]", encoding="utf-8")
    evidence = cw.build_evidence(run_dir, prompt="Who killed JFK?")
    (tmp_path / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    (tmp_path / "llm-retrieval-check.md").write_text("# LLM Retrieval Check\n", encoding="utf-8")
    (tmp_path / "llm-retrieval-check.json").write_text("{}", encoding="utf-8")

    found = cw.collect_artifacts(tmp_path, run_dir, "snapshot")
    run = cw.PromptRun(
        index=1,
        prompt="Who killed JFK?",
        slug="01_who_killed_jfk",
        run_id="ord__01",
        artifacts=found,
    )
    spec = cw.OrderSpec(
        order_id="ord",
        prompts=["Who killed JFK?"],
        product="snapshot",
        product_id="moyo_snapshot",
    )
    cw.write_prompt_report_json(tmp_path, spec, run, evidence=evidence)
    found = run.artifacts
    assert set(cw.CONTRACT_ARTIFACTS) <= set(found)
    report = json.loads(found["report.json"].read_text(encoding="utf-8"))
    assert report["orderId"] == "ord"
    assert report["productId"] == "moyo_snapshot"
    assert report["findings"][0]["claim"] == "fact"
    assert report["citations"] == ["https://example.com/source"]
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


def test_storage_destinations_single_prompt_writes_flat_qc_path_only(tmp_path: Path):
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
    assert dest == {"reports/ord/report.pdf": pdf}


def test_storage_destinations_multi_prompt_uses_slug_folders(tmp_path: Path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"%PDF")
    b.write_bytes(b"%PDF")
    runs = [
        cw.PromptRun(
            index=1,
            prompt="Enron",
            slug="01_enron",
            run_id="ord__01_enron",
            artifacts={"report.pdf": a},
        ),
        cw.PromptRun(
            index=2,
            prompt="Other",
            slug="02_other",
            run_id="ord__02_other",
            artifacts={"report.pdf": b},
        ),
    ]
    dest = dict(cw.storage_destinations("ord", runs))
    assert dest == {
        "reports/ord/01_enron/report.pdf": a,
        "reports/ord/02_other/report.pdf": b,
    }


def test_note_explore_gaps_when_empty(tmp_path: Path):
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
    notes = cw.note_explore_gaps(tmp_path, "Enron?")
    assert notes
    assert "0/2 usable" in notes[0]


def test_note_explore_gaps_partial_failures_does_not_raise(tmp_path: Path):
    (tmp_path / "raw_responses.json").write_text(
        json.dumps(
            [
                {"source_label": "GPT", "text": "Enron hid debt via SPEs."},
                {"source_label": "Grok", "error": "Connection error.", "text": ""},
            ]
        ),
        encoding="utf-8",
    )
    notes = cw.note_explore_gaps(tmp_path, "Enron?")
    assert notes
    assert "1/2 usable" in notes[0]
    assert "Grok" in notes[0]


def test_note_explore_gaps_ok(tmp_path: Path):
    (tmp_path / "raw_responses.json").write_text(
        json.dumps([{"source_label": "GPT", "text": "Enron hid debt via SPEs."}]),
        encoding="utf-8",
    )
    assert cw.note_explore_gaps(tmp_path, "Enron?") == []


def test_note_report_gaps_when_empty(tmp_path: Path):
    (tmp_path / "claims.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "chunks.jsonl").write_text("", encoding="utf-8")
    notes = cw.note_report_gaps(tmp_path, "Enron?")
    assert notes
    assert "0 claims" in notes[0]


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
    assert "reports/ord_x/01_enron/llm-retrieval-check.md" not in paths
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
    assert cfg["extract"]["provider"] == "custom"
    assert cfg["extract"]["model"] == "google/gemini-2.5-flash"
    assert "api_key" not in cfg["extract"]
    assert "aiplatform.googleapis.com" in (cfg["extract"].get("base_url") or "")
    assert cfg["extract"].get("prompt") == "prompts/extract_claims.md"
    assert cfg["synthesize"]["provider"] == "custom"
    assert cfg["synthesize"]["model"] == "google/gemini-2.5-flash"
    assert "api_key" not in cfg["synthesize"]
    assert "aiplatform.googleapis.com" in (cfg["synthesize"].get("base_url") or "")
    assert cfg["cluster"]["provider"] == "custom"
    assert cfg["cluster"]["model"] == "google/gemini-2.5-flash"
    assert "api_key" not in cfg["cluster"]
    assert "aiplatform.googleapis.com" in (cfg["cluster"].get("base_url") or "")
    assert "sk-test-openai" not in dest.read_text(encoding="utf-8")
    assert cfg["render"]["headline"] == "Cover"


def test_output_paths_prefer_canonical_root():
    urls = {
        "reports/ord/01_enron/report.pdf": "gs://b/reports/ord/01_enron/report.pdf",
        "reports/ord/report.pdf": "gs://b/reports/ord/report.pdf",
        "reports/ord/report.json": "gs://b/reports/ord/report.json",
        "reports/ord/report.md": "gs://b/reports/ord/report.md",
        "reports/ord/report.html": "gs://b/reports/ord/report.html",
    }
    out = cw.output_paths("ord", urls)
    assert out["pdfPath"] == "reports/ord/report.pdf"
    assert out["jsonPath"] == "reports/ord/report.json"
    assert out["markdownPath"] == "reports/ord/report.md"
    assert out["htmlPath"] == "reports/ord/report.html"


def test_success_update_fields_awaiting_qc_when_qc_required():
    spec = cw.OrderSpec(order_id="ord", prompts=["Enron"], qc_required=True)
    urls = {"reports/ord/report.pdf": "gs://b/reports/ord/report.pdf"}
    fields = cw.success_update_fields(
        spec,
        started="2026-08-18T00:00:00+00:00",
        finished="2026-08-18T01:00:00+00:00",
        urls=urls,
        manifest={"orderId": "ord"},
    )
    assert fields["reportStatus"] == "awaiting_qc"
    assert fields["qcStatus"] == "pending"
    assert fields["qcRequired"] is True
    assert "deliveredAt" not in fields
    assert fields["output"]["pdfPath"] == "reports/ord/report.pdf"
    assert fields["reportStatus"] != "qc_pending"


def test_success_update_fields_delivered_when_qc_not_required():
    spec = cw.OrderSpec(
        order_id="ord",
        prompts=["Enron"],
        qc_required=False,
        generation_mode="full",
    )
    fields = cw.success_update_fields(
        spec,
        started="2026-08-18T00:00:00+00:00",
        finished="2026-08-18T01:00:00+00:00",
        urls={"reports/ord/report.json": "gs://b/reports/ord/report.json"},
        manifest={"orderId": "ord"},
    )
    assert fields["reportStatus"] == "delivered"
    assert fields["qcStatus"] == "not_required"
    assert fields["deliveredAt"] == "2026-08-18T01:00:00+00:00"
    assert fields["generationMode"] == "full"
    assert fields["output"]["jsonPath"] == "reports/ord/report.json"
    assert fields["qcRequired"] is False


def test_agent_report_json_is_machine_readable(tmp_path: Path):
    evidence = {
        "headline": "Exposure",
        "topic": "Enron",
        "counts": {"findings": 1},
        "findings": [
            {
                "claim_id": "C1",
                "claim": "SPE debt was hidden",
                "citations": ["https://example.com/10k"],
                "source_models": ["GPT"],
            }
        ],
    }
    (tmp_path / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    spec = cw.OrderSpec(
        order_id="ord_agent",
        prompts=["Enron"],
        product="snapshot",
        product_id="moyo_snapshot",
        qc_required=False,
        source="stripe_mpp",
        generation_mode="full",
    )
    run = cw.PromptRun(
        1,
        "Enron",
        "01_enron",
        "ord_agent__01",
        {"evidence.json": tmp_path / "evidence.json"},
    )
    payload = cw.build_canonical_report(spec, [run])
    assert payload["orderId"] == "ord_agent"
    assert payload["productId"] == "moyo_snapshot"
    assert payload["product"] == "snapshot"
    assert payload["findings"][0]["claim"] == "SPE debt was hidden"
    assert payload["citations"] == ["https://example.com/10k"]
    dest = cw.write_canonical_report_json(spec, [run], tmp_path / "report.json")
    urls = {f"reports/{spec.order_id}/report.json": f"gs://b/reports/{spec.order_id}/report.json"}
    fields = cw.success_update_fields(
        spec,
        started="2026-08-18T00:00:00+00:00",
        finished="2026-08-18T01:00:00+00:00",
        urls=urls,
        manifest={"orderId": spec.order_id},
    )
    assert dest.is_file()
    assert fields["reportStatus"] == "delivered"
    assert fields["output"]["jsonPath"] == "reports/ord_agent/report.json"


def test_parse_order_snapshot_mpp_is_queued_shape():
    spec = cw.parse_order(
        "ord_snap",
        {
            "product": "snapshot",
            "productId": "moyo_snapshot",
            "prompts": ["What can models infer about Acme?"],
            "source": "stripe_mpp",
            "qcRequired": False,
            "generationMode": "full",
        },
    )
    assert spec.product == "snapshot"
    assert spec.product_id == "moyo_snapshot"
    assert spec.qc_required is False
    assert spec.generation_mode == "full"


def test_canonical_report_json_aggregates_prompts(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "evidence.json").write_text(
        json.dumps(
            {
                "headline": "A",
                "findings": [{"claim_id": "C1", "claim": "one", "citations": ["s1"]}],
            }
        ),
        encoding="utf-8",
    )
    (b / "evidence.json").write_text(
        json.dumps(
            {
                "headline": "B",
                "findings": [{"claim_id": "C2", "claim": "two", "citations": ["s2"]}],
            }
        ),
        encoding="utf-8",
    )
    spec = cw.OrderSpec(
        order_id="ord",
        prompts=["Alpha", "Beta"],
        product="basis",
        product_id="moyo_basis",
        generation_mode="full",
    )
    runs = [
        cw.PromptRun(1, "Alpha", "01_alpha", "ord__01", {"evidence.json": a / "evidence.json"}),
        cw.PromptRun(2, "Beta", "02_beta", "ord__02", {"evidence.json": b / "evidence.json"}),
    ]
    payload = cw.build_canonical_report(spec, runs, generated_at="2026-08-18T00:00:00+00:00")
    assert payload["orderId"] == "ord"
    assert payload["productId"] == "moyo_basis"
    assert payload["counts"]["reports"] == 2
    assert [f["claim"] for f in payload["findings"]] == ["one", "two"]
    assert payload["citations"] == ["s1", "s2"]
    assert len(payload["reports"]) == 2

