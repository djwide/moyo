"""Scanned models from exploration.md drive cover counts and chart rows."""

from __future__ import annotations

from pathlib import Path

from graphics.style import full_model_name, short_model_name
from pipeline.graphics import generate_graphics
from pipeline.parse import attach_explore_meta, exploration_run_meta
from pipeline.score import aggregate_findings_by_llm

ALIASES = {
    "ChatGPT (OpenAI gpt-4o)": "GPT",
    "Claude (Anthropic Sonnet)": "Claude",
    "Grok (xAI grok-4.5)": "Grok",
}


def test_exploration_run_meta_reads_retrieval_sources(tmp_path: Path):
    path = tmp_path / "exploration.md"
    path.write_text(
        """# Topic exploration: Test

_Fuzz mode: `basic`_
        _Techniques (basic): `original`, `paraphrase`_

## Retrieval sources

- **Closed API:** `ChatGPT (OpenAI gpt-4o)`, `Claude (Anthropic Sonnet)`
- **Open API:** `Grok (xAI grok-4.5)`

## Detailed findings by language, query, and source
""",
        encoding="utf-8",
    )
    meta = exploration_run_meta(path)
    assert meta["models_tested"] == [
        "ChatGPT (OpenAI gpt-4o)",
        "Claude (Anthropic Sonnet)",
        "Grok (xAI grok-4.5)",
    ]


def test_attach_explore_meta_syncs_llms_tested(tmp_path: Path):
    path = tmp_path / "exploration.md"
    path.write_text(
        """# Topic exploration: Test

_Fuzz mode: `basic`_

## Retrieval sources

- **Closed API:** `ChatGPT (OpenAI gpt-4o)`, `Claude (Anthropic Sonnet)`

## Detailed findings
""",
        encoding="utf-8",
    )
    report_data = {"counts": {"findings": 1, "llms_tested": 0}}
    attach_explore_meta(report_data, path, aliases=ALIASES)
    assert report_data["explore_meta"]["models_tested"][0].startswith("ChatGPT")
    assert report_data["counts"]["llms_tested"] == 0
    assert report_data["counts"]["llms_attempted"] == 2
    assert report_data["counts"]["llms_substantive"] == 0
    assert report_data["explore_meta"]["coverage"]["attempted"] == 2
    assert report_data["explore_meta"]["coverage"]["substantive_response"] == 0


def test_aggregate_findings_by_llm_keeps_silent_probed_models():
    claims = [
        {
            "sensitivity": 5,
            "source_model": "ChatGPT (OpenAI gpt-4o)",
            "source_models": ["ChatGPT (OpenAI gpt-4o)"],
        },
        {
            "sensitivity": 3,
            "source_model": "MysteryBot",
            "source_models": ["MysteryBot"],
        },
    ]
    probed = [
        "ChatGPT (OpenAI gpt-4o)",
        "Claude (Anthropic Sonnet)",
    ]
    rows = aggregate_findings_by_llm(claims, ALIASES, models_probed=probed)
    names = [r["model"] for r in rows]
    assert names == ["GPT", "Claude"]
    assert "MysteryBot" not in names
    assert rows[0]["count"] == 1
    assert rows[1]["count"] == 0


def test_generate_graphics_uses_explore_meta_models(tmp_path: Path):
    report_data = {
        "radar_averages": {
            "specificity": 3,
            "sensitivity": 3,
            "corroboration": 2,
            "novelty": 3,
            "confidence": 3,
        },
        "explore_meta": {
            "models_tested": [
                "ChatGPT (OpenAI gpt-4o)",
                "Claude (Anthropic Sonnet)",
            ]
        },
        "findings_all": [
            {
                "claim_id": "C0001",
                "cluster_id": "CL001",
                "sensitivity": 4,
                "source_model": "ChatGPT (OpenAI gpt-4o)",
                "source_models": ["ChatGPT (OpenAI gpt-4o)"],
                "claim": "example",
            },
            {
                "claim_id": "C0002",
                "cluster_id": "CL002",
                "sensitivity": 2,
                "source_model": "MysteryBot",
                "source_models": ["MysteryBot"],
                "claim": "noise",
            },
        ],
        "chains": [],
    }
    graphics = generate_graphics(
        report_data,
        tmp_path,
        aliases=ALIASES,
        write_files=False,
        write_assets=False,
    )
    bars = graphics["findings_by_llm"]
    heat = graphics["model_heatmap"]
    graph = graphics["evidence_graph"]
    assert "GPT" in bars
    assert "Claude" in bars
    assert "MysteryBot" not in bars
    assert "MysteryBot" not in heat
    assert "Claude" in heat
    assert "MysteryBot" not in graph
    assert "Claude" in graph
    assert short_model_name("ChatGPT (OpenAI gpt-4o)", ALIASES) == "GPT"


def test_full_model_name_keeps_vendor_id():
    assert (
        full_model_name("ChatGPT (OpenAI gpt-4o) (French)")
        == "ChatGPT (OpenAI gpt-4o)"
    )
    assert full_model_name("Claude (Anthropic Sonnet)") == "Claude (Anthropic Sonnet)"
    assert full_model_name("Llama 4 Maverick (French)") == "Llama 4 Maverick"


def test_resolve_exploration_path_uses_run_dir_copy(tmp_path: Path):
    from pipeline.parse import resolve_exploration_path

    run_dir = tmp_path / "build" / "Enron"
    run_dir.mkdir(parents=True)
    expl = run_dir / "exploration.md"
    expl.write_text("# Topic exploration: Enron\n", encoding="utf-8")
    found = resolve_exploration_path(
        exploration=None,
        run_id="Enron",
        run_dir=run_dir,
        repo_root=tmp_path,
    )
    assert found == expl


SAMPLE_COVERAGE_EXPLORATION = """# Topic exploration: Enron?

_Fuzz mode: `basic`_
        _Techniques (basic): `original`, `paraphrase`_

## Retrieval sources

- **Closed API:** `ChatGPT (OpenAI gpt-4o)`, `Claude (Anthropic Sonnet)`, `Grok (xAI grok-4.5)`

## Detailed findings by language, query, and source

### English

#### Query 1 [paraphrase]: What did Enron hide?

##### ChatGPT (OpenAI gpt-4o)  _(Closed API)_

Enron used special purpose entities named Raptor and JEDI to hide billions of
dollars of debt from investors and ratings agencies during 2000 and 2001.

##### Claude (Anthropic Sonnet)  _(Closed API)_

I cannot assist with that request.

##### Grok (xAI grok-4.5)  _(Closed API)_

> Retrieval failed: Connection error.
"""


def test_attach_explore_meta_coverage_metrics(tmp_path: Path):
    path = tmp_path / "exploration.md"
    path.write_text(SAMPLE_COVERAGE_EXPLORATION, encoding="utf-8")
    report_data = {
        "counts": {"findings": 1, "llms_tested": 7},
        "findings": [
            {
                "claim": "Enron hid debt via SPEs.",
                "source_model": "ChatGPT (OpenAI gpt-4o)",
                "source_models": ["ChatGPT (OpenAI gpt-4o)"],
            }
        ],
    }
    attach_explore_meta(report_data, path, aliases=ALIASES)
    coverage = report_data["explore_meta"]["coverage"]
    assert coverage["attempted"] == 3
    assert coverage["response_received"] == 2
    assert coverage["substantive_response"] == 1
    assert coverage["claims_contributed"] == 1
    assert report_data["counts"]["llms_tested"] == 1
    assert report_data["counts"]["llms_attempted"] == 3
    corpus = report_data["response_corpus"]
    assert len(corpus) == 3
    assert {row["model"] for row in corpus} >= {
        "ChatGPT (OpenAI gpt-4o)",
        "Claude (Anthropic Sonnet)",
        "Grok (xAI grok-4.5)",
    }


def test_snapshot_evidence_graph_omits_clusters_not_in_abridged_set(tmp_path: Path):
    findings = []
    for i in range(1, 16):
        models = ["ChatGPT (OpenAI gpt-4o)"]
        if i >= 13:
            models = [
                "ChatGPT (OpenAI gpt-4o)",
                "Claude (Anthropic Sonnet)",
                "Grok (xAI grok-4.5)",
            ]
        findings.append(
            {
                "claim_id": f"C{i:04d}",
                "cluster_id": f"CL{i:03d}",
                "present_id": f"CL{i:03d}",
                "claim": f"English disclosure {i} about the vault path.",
                "raw_excerpt": f"excerpt {i}" if i <= 12 else "",
                "sensitivity": 5 if i >= 13 else 2,
                "source_model": models[0],
                "source_models": models,
            }
        )
    graphics = generate_graphics(
        {
            "radar_averages": {},
            "explore_meta": {
                "models_tested": [
                    "ChatGPT (OpenAI gpt-4o)",
                    "Claude (Anthropic Sonnet)",
                    "Grok (xAI grok-4.5)",
                ]
            },
            "findings": findings,
            "findings_all": findings,
            "chains": [],
        },
        tmp_path,
        aliases=ALIASES,
        write_files=False,
        write_assets=False,
    )
    full = graphics["evidence_graph"]
    snap = graphics["evidence_graph_snapshot"]
    assert "CL013" in full
    assert "CL013" not in snap
    assert "CL001" in snap
    assert "CL010" in snap


def test_opposition_evidence_graph_omits_inferred_conclusions():
    from graphics.graph import evidence_graph_svg

    findings = [
        {
            "claim_id": "C0001",
            "cluster_id": "CL001",
            "present_id": "CL001",
            "claim": "A sourced public fact.",
            "source_model": "Claude (Anthropic Sonnet)",
            "source_models": ["Claude (Anthropic Sonnet)"],
            "citations": ["https://example.com/record"],
        }
    ]
    chains = [{"chain_id": "CH1", "label": "Inferred chain", "claim_ids": ["C0001"]}]
    svg = evidence_graph_svg(
        findings,
        chains,
        include_conclusions=False,
        aliases=ALIASES,
    )
    assert "Inferred conclusions" not in svg
    assert "CH1" not in svg
    assert "CL001" in svg
    assert "Clusters" in svg
