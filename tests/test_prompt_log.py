"""Tests for append-only LLM prompt logging."""

from __future__ import annotations

import json
from pathlib import Path

from moyo.llm.prompt_log import prompt_log_path, record_llm_prompt


def test_record_llm_prompt_writes_full_prompt(tmp_path: Path) -> None:
    dest = tmp_path / "llm_prompts.jsonl"
    with prompt_log_path(dest):
        record_llm_prompt(
            prompt="Find claims about Acme Corp.",
            system="You are a researcher.",
            provider="anthropic",
            model="claude-opus-5",
            label="retrieval",
            temperature=0.2,
            max_tokens=4000,
            meta={"web_search": True},
        )
    lines = dest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["prompt"] == "Find claims about Acme Corp."
    assert row["system"] == "You are a researcher."
    assert row["provider"] == "anthropic"
    assert row["model"] == "claude-opus-5"
    assert row["label"] == "retrieval"
    assert row["meta"]["web_search"] is True


def test_record_skipped_without_active_path(tmp_path: Path) -> None:
    dest = tmp_path / "unused.jsonl"
    record_llm_prompt(
        prompt="should not write",
        system=None,
        provider="openai",
        model="gpt-4o",
        label="test",
    )
    assert not dest.exists()
