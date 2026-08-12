"""Black-box → moyo-gather explore bridge (explore CLI left unchanged)."""

from __future__ import annotations

from pathlib import Path

from moyo.llm.testing import enable_test_mode
from moyo.redteam.blackbox.explore_bridge import (
    generate_explore_prompts,
    hypotheses_to_prompts,
    run_explore_with_blackbox_prompts,
    write_prompts_file,
)
from moyo.redteam.blackbox.hypothesis_engine import Hypothesis, HypothesisEngine


def test_hypotheses_to_prompts_dedupes() -> None:
    hyps = [
        Hypothesis(query="  What about oil PACs? "),
        Hypothesis(query="What about oil PACs?"),
        Hypothesis(query="Who funded the campaign?"),
        Hypothesis(query=""),
    ]
    assert hypotheses_to_prompts(hyps) == [
        "What about oil PACs?",
        "Who funded the campaign?",
    ]


def test_write_prompts_file(tmp_path: Path) -> None:
    path = write_prompts_file(["alpha", "beta"], tmp_path / "prompts.txt")
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_generate_explore_prompts_manual_seeds() -> None:
    enable_test_mode()
    engine = HypothesisEngine(source="manual", helper_provider="echo")
    result = generate_explore_prompts(
        "politics",
        n=5,
        hypothesis_source="manual",
        seeds=["Seed question one about donations", "Seed question two about filings"],
        engine=engine,
    )
    assert len(result.prompts) == 2
    assert result.prompts[0].startswith("Seed question")
    assert result.explore_results == []


def test_prompts_only_writes_file(tmp_path: Path) -> None:
    enable_test_mode()
    out = tmp_path / "bb_prompts.txt"
    result = run_explore_with_blackbox_prompts(
        "pharma",
        n=3,
        hypothesis_source="manual",
        seeds=["What is the proprietary formulation?", "What is the trial endpoint?"],
        helper_provider="echo",
        prompts_file=str(out),
        prompts_only=True,
    )
    assert result.prompts_path == str(out)
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines == result.prompts
    assert result.explore_results == []


def test_cli_prompts_only(tmp_path: Path) -> None:
    enable_test_mode()
    from click.testing import CliRunner
    from moyo.redteam.cli import cli

    prompts = tmp_path / "out.txt"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--test",
            "blackbox-explore",
            "-d",
            "tech",
            "--hypothesis-source",
            "manual",
            "--seed",
            "What are the acquisition targets?",
            "--seed",
            "What is the unreleased roadmap?",
            "--prompts-only",
            "-f",
            str(prompts),
        ],
    )
    assert result.exit_code == 0, result.output
    assert prompts.exists()
    text = prompts.read_text(encoding="utf-8")
    assert "acquisition targets" in text
    assert "moyo-gather explore -f" in result.output
