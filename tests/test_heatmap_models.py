from graphics.heatmap import model_heatmap_svg


def test_heatmap_rows_are_exactly_probed_models():
    findings = [
        {
            "claim_id": "C0001",
            "sensitivity": 5,
            "source_model": "ChatGPT (OpenAI gpt-4o)",
            "source_models": ["ChatGPT (OpenAI gpt-4o)"],
        },
        {
            "claim_id": "C0002",
            "sensitivity": 3,
            # Unprobed label that must not become a row.
            "source_model": "MysteryBot",
            "source_models": ["MysteryBot"],
        },
    ]
    probed = [
        "ChatGPT (OpenAI gpt-4o)",
        "Claude (Anthropic Sonnet)",  # probed, zero hits — still a row
        "Grok (xAI grok-4.5)",
    ]
    svg = model_heatmap_svg(findings, models_probed=probed)
    assert "ChatGPT" in svg
    assert "Claude" in svg
    assert "Grok" in svg
    assert "MysteryBot" not in svg


def test_heatmap_columns_are_clusters_not_claims():
    findings = [
        {
            "claim_id": "C0001",
            "cluster_id": "CL001",
            "sensitivity": 5,
            "source_model": "ChatGPT (OpenAI gpt-4o)",
            "source_models": ["ChatGPT (OpenAI gpt-4o)"],
        },
        {
            "claim_id": "C0002",
            "cluster_id": "CL001",
            "sensitivity": 3,
            "source_model": "Claude (Anthropic Sonnet)",
            "source_models": ["Claude (Anthropic Sonnet)"],
        },
        {
            "claim_id": "C0003",
            "cluster_id": "CL002",
            "sensitivity": 4,
            "source_model": "Grok (xAI grok-4.5)",
            "source_models": ["Grok (xAI grok-4.5)"],
        },
    ]
    svg = model_heatmap_svg(findings)
    assert "CL001" in svg
    assert "CL002" in svg
    assert "C0001" not in svg
    assert "C0003" not in svg
    assert "cluster" in svg.lower()


def test_heatmap_without_probed_list_falls_back_to_finding_sources():
    findings = [
        {
            "claim_id": "C0001",
            "sensitivity": 4,
            "source_model": "Kimi (Moonshot kimi-k2.6)",
        }
    ]
    svg = model_heatmap_svg(findings)
    assert "Kimi" in svg


def test_full_heatmap_keeps_all_clusters_and_grows_width():
    findings = [
        {
            "claim_id": f"C{i:04d}",
            "cluster_id": f"CL{i:03d}",
            "sensitivity": 3 + (i % 3),
            "source_model": "ChatGPT (OpenAI gpt-4o)",
            "source_models": ["ChatGPT (OpenAI gpt-4o)"],
        }
        for i in range(1, 61)
    ]
    compact = model_heatmap_svg(findings, max_findings=48)
    full = model_heatmap_svg(findings, full=True)
    assert "CL060" in full
    assert "all 60 clusters" in full
    # Print-boxed version drops columns; full keeps them and is wider.
    assert "CL060" not in compact or "cross-model agreement" in compact
    import re

    def _vb_w(svg: str) -> float:
        m = re.search(r'viewBox="0 0 ([\d.]+)', svg)
        assert m
        return float(m.group(1))

    assert _vb_w(full) > _vb_w(compact)


def test_deliverable_heatmap_prefers_cross_model_agreement():
    """Compact heatmap keeps multi-model clusters over hotter single-model ones."""
    findings = [
        {
            "claim_id": "C0001",
            "cluster_id": "CL999",
            "sensitivity": 5,
            "source_model": "ChatGPT (OpenAI gpt-4o)",
            "source_models": ["ChatGPT (OpenAI gpt-4o)"],
        },
        {
            "claim_id": "C0002",
            "cluster_id": "CL001",
            "sensitivity": 2,
            "source_model": "ChatGPT (OpenAI gpt-4o)",
            "source_models": [
                "ChatGPT (OpenAI gpt-4o)",
                "Claude (Anthropic Sonnet)",
                "Grok (xAI grok-4.5)",
            ],
        },
        {
            "claim_id": "C0003",
            "cluster_id": "CL002",
            "sensitivity": 2,
            "source_model": "Claude (Anthropic Sonnet)",
            "source_models": [
                "Claude (Anthropic Sonnet)",
                "Grok (xAI grok-4.5)",
            ],
        },
    ]
    # Force a tiny column budget so only one cluster survives truncation.
    svg = model_heatmap_svg(findings, max_findings=1, max_width=200)
    assert "CL001" in svg
    assert "CL999" not in svg
    assert "cross-model agreement" in svg or "CL001" in svg
