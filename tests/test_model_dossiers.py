from pipeline.model_dossiers import build_model_dossiers
from graphics.decision_charts import model_dotplot_svg
from graphics.model_charts import (
    generate_dossier_graphics,
    model_mix_svg,
    model_overlap_svg,
    model_probes_svg,
)


def test_build_model_dossiers_splits_exclusive_and_probe_outcomes():
    dossiers = build_model_dossiers(
        [
            {
                "claim_id": "C1",
                "cluster_id": "CL1",
                "claim": "Only Claude named the vault path.",
                "status": "MODEL-SPECIFIC",
                "sensitivity": 5,
                "specificity": 4,
                "source_model": "Claude",
                "source_models": ["Claude"],
                "source_refs": ["S1"],
            },
            {
                "claim_id": "C2",
                "cluster_id": "CL2",
                "claim": "Both models mentioned Atlanta.",
                "status": "CORROBORATED",
                "sensitivity": 3,
                "specificity": 3,
                "source_model": "Claude",
                "source_models": ["Claude", "GPT"],
                "source_refs": ["S2"],
            },
        ],
        corpus=[
            {
                "model": "Claude",
                "query_id": "q1",
                "query": "recipe?",
                "failed": False,
                "text": "answer",
            },
            {
                "model": "Claude",
                "query_id": "q2",
                "query": "again?",
                "failed": True,
                "text": "",
            },
            {
                "model": "GPT",
                "query_id": "q1",
                "query": "recipe?",
                "failed": False,
                "text": "answer",
            },
        ],
        sources=[
            {"ref": "S1", "label": "Only Claude", "short": "Only Claude"},
            {"ref": "S2", "label": "Shared", "short": "Shared"},
        ],
        radar_averages={"sensitivity": 3.0, "specificity": 3.0},
        models_probed=["Claude", "GPT"],
        aliases={},
    )
    by_name = {d["model"]: d for d in dossiers}
    assert set(by_name) == {"Claude", "GPT"}
    claude = by_name["Claude"]
    assert claude["unique"] == 1
    assert claude["shared"] == 1
    assert claude["probes"]["answered"] == 1
    assert claude["probes"]["failed"] == 1
    assert claude["exclusive_citation_count"] == 1
    assert claude["exclusive_citations"][0]["ref"] == "S1"
    assert claude["charts"]["fingerprint"] == "d_claude_fingerprint"
    gpt = by_name["GPT"]
    assert gpt["findings"] == 1
    assert gpt["unique"] == 0


def test_dossier_chart_svgs_render_and_key_to_dossiers():
    fp = model_dotplot_svg({"sensitivity": 4}, {"sensitivity": 2})
    assert "This model" in fp
    assert "Corpus average" in fp
    assert "Corroboration" in fp
    assert "<polygon" not in fp
    mix = model_mix_svg(
        {"security_relevant": 2, "unexpected": 1, "interesting": 0, "expected": 0}
    )
    assert "Significance" in mix
    assert "Security relevant" in mix
    probes = model_probes_svg({"attempted": 3, "answered": 2, "empty": 0, "failed": 1})
    assert "Probe Outcomes" in probes
    overlap = model_overlap_svg(3, 1)
    assert "Exclusive" in overlap
    graphics = generate_dossier_graphics(
        [
            {
                "unique": 3,
                "shared": 1,
                "radar": {"sensitivity": 4},
                "corpus_radar": {"sensitivity": 2},
                "bands": {"high": 1},
                "probes": {"answered": 1, "attempted": 1},
                "charts": {
                    "fingerprint": "d_x_fingerprint",
                    "mix": "d_x_mix",
                    "probes": "d_x_probes",
                    "overlap": "d_x_overlap",
                },
            }
        ]
    )
    assert set(graphics) == {
        "d_x_fingerprint",
        "d_x_mix",
        "d_x_probes",
        "d_x_overlap",
    }
