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
