from moyo.gui.cloud_compute import (
    CloudComputeConfig,
    build_order_payload,
    firestore_document,
    new_gui_order_id,
)


def test_build_order_payload_requires_prompt():
    try:
        build_order_payload(prompts=["  "])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_build_order_payload_shape():
    payload = build_order_payload(
        prompts=["What happened at Enron?"],
        product="both",
        fuzz_mode="multilingual",
        strategies=["paraphrase", "abstract"],
        languages=["Spanish"],
        include_remediation=True,
        seeds=2,
    )
    assert payload["prompts"] == ["What happened at Enron?"]
    assert payload["product"] == "both"
    assert payload["paymentStatus"] == "paid"
    assert payload["reportStatus"] == "queued"
    assert payload["qcRequired"] is True
    assert payload["qcStatus"] == "pending"
    assert payload["source"] == "gui"
    assert payload["seeds"] == 2
    assert payload["includeRemediation"] is True


def test_firestore_value_encodes_lists_and_bools():
    doc = firestore_document(
        {"prompts": ["a", "b"], "paid": True, "seeds": 3, "empty": None}
    )
    fields = doc["fields"]
    assert fields["paid"] == {"booleanValue": True}
    assert fields["seeds"] == {"integerValue": "3"}
    assert fields["empty"] == {"nullValue": None}
    values = fields["prompts"]["arrayValue"]["values"]
    assert values[0] == {"stringValue": "a"}


def test_new_gui_order_id_prefix():
    oid = new_gui_order_id()
    assert oid.startswith("ord_gui_")


def test_cloud_config_from_env(monkeypatch):
    monkeypatch.setenv("MOYO_CLOUD_PROJECT", "demo-proj")
    monkeypatch.setenv("MOYO_CLOUD_REGION", "us-east1")
    monkeypatch.setenv("MOYO_CLOUD_JOB", "custom-job")
    cfg = CloudComputeConfig.from_env()
    assert cfg.project == "demo-proj"
    assert cfg.region == "us-east1"
    assert cfg.job == "custom-job"
