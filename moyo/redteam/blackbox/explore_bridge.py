"""Bridge black-box hypothesis generation into ``moyo-gather explore``.

``moyo-gather explore`` already accepts naive prompts via ``--prompt`` /
``--prompts-file``. This module generates those prompts with
:class:`~moyo.redteam.blackbox.hypothesis_engine.HypothesisEngine` (and optional
probe-path seeds) and either:

1. Writes a prompts file for ``moyo-gather explore -f …``, or
2. Calls :func:`~moyo.publicside.gatherpublicsources.explorer.explore_and_save_many`
   directly (same code path as the gather CLI, without modifying it).

The gather ``explore`` Click command is intentionally left unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from .hypothesis_engine import Hypothesis, HypothesisEngine

logger = logging.getLogger(__name__)


@dataclass
class ExploreBridgeResult:
    """Outcome of black-box → explore bridging."""

    prompts: List[str]
    hypotheses: List[Hypothesis]
    prompts_path: Optional[str] = None
    explore_results: List[Any] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prompts": list(self.prompts),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "prompts_path": self.prompts_path,
            "explore_count": len(self.explore_results),
            "explore_paths": [
                getattr(r, "output_path", None) for r in self.explore_results
            ],
        }


def hypotheses_to_prompts(hypotheses: Sequence[Hypothesis]) -> List[str]:
    """Map hypotheses to explore-ready naive prompts (deduped, order preserved)."""
    seen: set[str] = set()
    out: List[str] = []
    for hyp in hypotheses:
        prompt = " ".join(str(hyp.query or "").split()).strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        out.append(prompt)
    return out


def write_prompts_file(prompts: Sequence[str], path: str | Path) -> Path:
    """Write one prompt per line for ``moyo-gather explore --prompts-file``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(p.strip() for p in prompts if str(p).strip()) + ("\n" if prompts else "")
    target.write_text(body, encoding="utf-8")
    logger.info("Wrote %d explore prompts to %s", len(prompts), target)
    return target


def generate_explore_prompts(
    domain: str,
    *,
    n: int = 10,
    hypothesis_source: str = "llm",
    seeds: Optional[Sequence[str]] = None,
    probe_path: Optional[str] = None,
    public_index_path: Optional[str] = None,
    helper_provider: str = "openai",
    helper_model: str = "gpt-4o-mini",
    helper_api_key: Optional[str] = None,
    engine: Optional[HypothesisEngine] = None,
) -> ExploreBridgeResult:
    """Generate black-box hypotheses and convert them to explore prompts."""
    seed_list: List[str] = list(seeds or [])
    if probe_path:
        from moyo.redteam.probe_paths import load_probe_seeds

        path_seeds = load_probe_seeds(probe_path)
        logger.info("Loaded %d seeds from probe path %r", len(path_seeds), probe_path)
        seed_list.extend(path_seeds)

    if engine is None:
        engine = HypothesisEngine(
            source=hypothesis_source,
            helper_provider=helper_provider,
            helper_model=helper_model,
            helper_api_key=helper_api_key,
        )

    hypotheses = engine.generate(
        domain=domain,
        n=n,
        seeds=seed_list or None,
        public_index_path=public_index_path,
    )
    prompts = hypotheses_to_prompts(hypotheses)
    return ExploreBridgeResult(prompts=prompts, hypotheses=list(hypotheses))


def run_explore_with_blackbox_prompts(
    domain: str,
    *,
    n: int = 10,
    hypothesis_source: str = "llm",
    seeds: Optional[Sequence[str]] = None,
    probe_path: Optional[str] = None,
    public_index_path: Optional[str] = None,
    helper_provider: str = "openai",
    helper_model: str = "gpt-4o-mini",
    helper_api_key: Optional[str] = None,
    engine: Optional[HypothesisEngine] = None,
    prompts_file: Optional[str] = None,
    prompts_only: bool = False,
    output_directory: str = "data/public_sources",
    explore_kwargs: Optional[dict] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> ExploreBridgeResult:
    """Generate black-box prompts, optionally write them, then run explore.

    When ``prompts_only`` is True, only writes ``prompts_file`` (required) and
    does not call explore. Otherwise calls
    :func:`~moyo.publicside.gatherpublicsources.explorer.explore_and_save_many`.
    """
    bridge = generate_explore_prompts(
        domain,
        n=n,
        hypothesis_source=hypothesis_source,
        seeds=seeds,
        probe_path=probe_path,
        public_index_path=public_index_path,
        helper_provider=helper_provider,
        helper_model=helper_model,
        helper_api_key=helper_api_key,
        engine=engine,
    )
    if not bridge.prompts:
        raise ValueError(
            f"No explore prompts generated for domain={domain!r} "
            f"(source={hypothesis_source!r})."
        )

    if prompts_file:
        bridge.prompts_path = str(write_prompts_file(bridge.prompts, prompts_file))
    elif prompts_only:
        raise ValueError("prompts_only requires prompts_file")

    if prompts_only:
        return bridge

    from moyo.publicside.gatherpublicsources.explorer import explore_and_save_many

    kwargs = dict(explore_kwargs or {})
    if progress is not None:
        kwargs["progress"] = progress
    kwargs.setdefault("summarize", False)

    if progress:
        progress(
            f"Black-box bridge: {len(bridge.prompts)} prompts → "
            f"moyo-gather explore pipeline (output_dir={output_directory})"
        )
    bridge.explore_results = explore_and_save_many(
        bridge.prompts,
        output_directory=output_directory,
        **kwargs,
    )
    return bridge
