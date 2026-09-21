from pipeline.content import build_content_doc
from pipeline.score import build_model_contrast, score_report


def test_build_model_contrast_splits_shared_and_unique():
    contrast = build_model_contrast(
        [
            {
                "claim_id": "C0001",
                "claim": "The vault path is public in a gist.",
                "status": "CORROBORATED",
                "source_models": ["ChatGPT", "Claude"],
            },
            {
                "claim_id": "C0002",
                "claim": "Only Grok named an unannounced SKU.",
                "status": "MODEL-SPECIFIC",
                "source_model": "Grok",
            },
            {
                "claim_id": "C0003",
                "claim": "Sources disagree on the launch window.",
                "status": "CONTESTED",
                "source_models": ["ChatGPT", "Gemini"],
            },
        ]
    )
    assert contrast["model_count"] == 4
    assert contrast["shared_count"] == 1
    assert contrast["unique_count"] == 1
    assert contrast["commonality"][0]["claim_id"] == "C0001"
    kinds = {row["kind"] for row in contrast["differences"]}
    assert "contested" in kinds
    assert "model-specific" in kinds
    assert "corroborated" in contrast["lede"].lower() or "across models" in contrast["lede"]


def test_score_report_includes_model_contrast():
    data = score_report(
        [
            {
                "claim_id": "C0001",
                "claim": "Shared public filing names the chipset.",
                "status": "CORROBORATED",
                "sensitivity": 4,
                "specificity": 4,
                "novelty": 2,
                "interestingness": 3,
                "confidence": 4,
                "source_models": ["ChatGPT", "Claude"],
            },
            {
                "claim_id": "C0002",
                "claim": "Grok alone cited an internal job code.",
                "status": "MODEL-SPECIFIC",
                "sensitivity": 5,
                "specificity": 5,
                "novelty": 4,
                "interestingness": 4,
                "confidence": 3,
                "source_model": "Grok",
            },
        ],
        [],
        run_id="t",
        topic="chipset",
        config={},
        graphics_cfg={},
    )
    contrast = data["model_contrast"]
    assert contrast["shared_count"] == 1
    assert contrast["unique_count"] == 1


def test_content_doc_exposes_model_contrast_page():
    doc = build_content_doc(
        {
            "run_id": "t",
            "topic": "Vault",
            "counts": {"findings": 2, "llms_tested": 2, "high_sensitivity": 1},
            "top_finding": {
                "claim_id": "C0001",
                "text": "Vault path is public.",
                "badges": [],
            },
            "findings": [
                {
                    "claim_id": "C0001",
                    "claim": "Vault path is public.",
                    "status": "CORROBORATED",
                    "sensitivity": 4,
                    "specificity": 4,
                    "source_models": ["ChatGPT", "Claude"],
                    "confidence": 4,
                },
                {
                    "claim_id": "C0002",
                    "claim": "Only Claude named a supplier SKU.",
                    "status": "MODEL-SPECIFIC",
                    "sensitivity": 3,
                    "specificity": 4,
                    "source_model": "Claude",
                    "confidence": 3,
                },
            ],
            "clusters": [],
        },
        report_date="20 Sep 2026",
    )
    assert doc["pages"]["model_comparison"]["title"] == "Model Comparison"
    assert doc["model_contrast"]["shared_count"] == 1
    assert doc["model_contrast"]["unique_count"] == 1
    assert doc["pages"]["model_comparison"]["body"]
