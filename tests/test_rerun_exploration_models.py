"""Overwrite selected-model retrieval in an existing exploration.md."""

from __future__ import annotations

from moyo.llm.client import LLMClient, LLMSpec
from moyo.publicside.gatherpublicsources.explorer import (
    RetrievalResult,
    merge_model_results,
    parse_exploration_markdown,
    rerun_exploration_models,
    selected_model_failures,
)

SAMPLE = """# Topic exploration: Test

_Generated 2026-09-21 12:00:00_
_Fuzz mode: `basic`_
_Techniques (basic): `original`, `paraphrase`_

## Retrieval sources

- **Closed API:** `ChatGPT (OpenAI gpt-4o)`, `Claude (Anthropic Sonnet)`

## Reworded query seeds

### English

1. `original` What did they hide?

## Detailed findings by language, query, and source

### English

#### Query 1 [original]: What did they hide?

##### ChatGPT (OpenAI gpt-4o)  _(Closed API)_

They hid debt via special purpose entities.

##### Claude (Anthropic Sonnet)  _(Closed API)_

> Retrieval failed: Connection error.
"""


def test_merge_model_results_overwrites_only_selected_label():
    corpus = parse_exploration_markdown(SAMPLE)
    replacement = RetrievalResult(
        seed="What did they hide?",
        llm_label="Claude (Anthropic Sonnet)",
        provider="anthropic",
        model="claude-sonnet-4-6",
        kind="closed",
        text="Claude now returns the SPE names.",
        seed_index=0,
        llm_index=1,
    )
    merge_model_results(corpus, [replacement], ["Claude (Anthropic Sonnet)"])
    texts = {row.llm_label: row.text for row in corpus.queries[0].results}
    assert "special purpose entities" in texts["ChatGPT (OpenAI gpt-4o)"]
    assert texts["Claude (Anthropic Sonnet)"] == "Claude now returns the SPE names."


def test_selected_model_failures_lists_empty_and_error_rows():
    rows = [
        RetrievalResult(
            seed="q",
            llm_label="Claude (Anthropic Sonnet)",
            provider="anthropic",
            model="x",
            kind="closed",
            text="",
            error="timeout",
            seed_index=0,
        )
    ]
    failures = selected_model_failures(rows, ["Claude (Anthropic Sonnet)", "Grok (xAI grok-4.5)"])
    assert any("timeout" in item for item in failures)
    assert any("Grok" in item for item in failures)


def test_rerun_exploration_models_overwrites_then_rebuild_gate(monkeypatch):
    from moyo.publicside.gatherpublicsources import explorer as expl

    async def fake_jobs(jobs, **kwargs):
        out = []
        for seed_index, qs, llm_index, llm in jobs:
            out.append(
                RetrievalResult(
                    seed=qs.text,
                    llm_label=llm.label,
                    provider=llm.spec.provider,
                    model=llm.spec.model,
                    kind=getattr(llm, "kind", None) or "closed",
                    text="Recovered after rerun.",
                    seed_index=seed_index,
                    llm_index=llm_index,
                )
            )
        return out

    monkeypatch.setattr(expl, "_retrieve_jobs_async", fake_jobs)
    monkeypatch.setattr(expl, "_run_coro", lambda coro: __import__("asyncio").run(coro))

    claude = LLMClient(
        LLMSpec(
            provider="anthropic",
            model="claude-sonnet-4-6",
            label="Claude (Anthropic Sonnet)",
            api_key="x",
        )
    )
    outcome = rerun_exploration_models(SAMPLE, [claude])
    assert outcome.ok
    assert "Recovered after rerun." in outcome.explore.markdown
    assert "They hid debt via special purpose entities." in outcome.explore.markdown
    assert "Retrieval failed: Connection error." not in outcome.explore.markdown
