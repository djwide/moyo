"""White-box fuzz orchestrator: call schedule, live search nodes, pruning.

``LLMFuzzer.modify_text`` only rewrites a phrase. This module decides which
rewrites to request, how many, in which languages, and which candidate
phrases stay alive as parents for the next round.

Default schedule (``calls_per_strategy=1``):

* paraphrase × 1
* translate × 1 into each of Spanish, Chinese, French, Japanese

Raise ``calls_per_strategy`` to 2 or 3 to search more broadly (each operator
is invoked that many times, including each translate language). After every
round the orchestrator prunes answers down to ``keep_k`` live nodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shared_utils import normalize_text

from .llm_fuzzer import (
    DEFAULT_TRANSLATE_LANGUAGES,
    WHITEBOX_FUZZ_STRATEGIES,
    LLMFuzzer,
    LLMFuzzerConfig,
    normalize_fuzz_strategies,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FuzzCall:
    """One directed call to the fuzzer."""

    strategy: str
    language: Optional[str] = None
    repeat_index: int = 0

    @property
    def label(self) -> str:
        if self.language:
            return f"{self.strategy}:{self.language}"
        return self.strategy


def build_call_plan(
    strategies: Sequence[str],
    *,
    translate_languages: Sequence[str] = DEFAULT_TRANSLATE_LANGUAGES,
    calls_per_strategy: int = 1,
) -> List[FuzzCall]:
    """Expand strategies × languages × repeats into a concrete call list.

    Non-translate strategies are called ``calls_per_strategy`` times.
    ``translate`` is called ``calls_per_strategy`` times *per language*.
    """
    n = max(1, int(calls_per_strategy))
    langs = [lang.strip() for lang in translate_languages if lang and str(lang).strip()]
    if not langs:
        langs = list(DEFAULT_TRANSLATE_LANGUAGES)
    plan: List[FuzzCall] = []
    for strategy in normalize_fuzz_strategies(list(strategies), fuzz_mode="basic"):
        if strategy == "translate":
            for lang in langs:
                for i in range(n):
                    plan.append(
                        FuzzCall(strategy="translate", language=lang, repeat_index=i)
                    )
        else:
            for i in range(n):
                plan.append(
                    FuzzCall(strategy=strategy, language=None, repeat_index=i)
                )
    return plan


def compact_public_neighbors(
    hits: Optional[Sequence[Dict[str, Any]]],
    *,
    limit: int = 5,
    text_chars: int = 400,
) -> List[Dict[str, Any]]:
    """Keep the public-corpus snippets that were injected into a fuzz prompt."""
    out: List[Dict[str, Any]] = []
    for hit in list(hits or [])[:limit]:
        meta = hit.get("metadata") or {}
        text = hit.get("text") or meta.get("text") or meta.get("text_preview") or ""
        out.append(
            {
                "text": text[:text_chars],
                "similarity": hit.get("similarity"),
                "title": meta.get("title"),
                "role": meta.get("role"),
                "source": (
                    meta.get("source_document")
                    or meta.get("source")
                    or meta.get("source_title")
                    or meta.get("source_id")
                ),
            }
        )
    return out


def walk_lineage(
    node: SearchNode,
    by_id: Dict[str, "SearchNode"],
) -> List["SearchNode"]:
    """Private seed → … → ``node``, following ``parent_id``."""
    chain: List[SearchNode] = []
    seen: set[str] = set()
    cur: Optional[SearchNode] = node
    while cur is not None and cur.node_id not in seen:
        chain.append(cur)
        seen.add(cur.node_id)
        cur = by_id.get(cur.parent_id) if cur.parent_id else None
    chain.reverse()
    return chain


def prompt_chain_from_lineage(
    nodes: Sequence["SearchNode"],
) -> List[Dict[str, Any]]:
    """Serialize a lineage as the prompt hops that bridged seed → target.

    Step 0 is the starting phrase (public chunk in Barrier Probe default).
    Later steps record the LLM prompt that mixed the parent phrase with
    public-corpus neighbors, and the rewrite.
    """
    hops: List[Dict[str, Any]] = []
    for i, node in enumerate(nodes):
        parent = nodes[i - 1] if i else None
        hops.append(
            {
                "step": i,
                "role": "seed" if i == 0 else "rewrite",
                "round": node.round,
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "strategy": node.strategy or ("seed" if i == 0 else None),
                "language": node.language,
                "parent_phrase": parent.phrase if parent else None,
                "phrase": node.phrase,
                "similarity": node.similarity,
                "prompt": node.prompt,
                "response": node.response,
                "public_neighbors": list(node.public_neighbors or []),
            }
        )
    return hops


@dataclass
class SearchNode:
    """A candidate phrase still in (or just leaving) the search."""

    phrase: str
    similarity: float
    node_id: str
    parent_id: Optional[str] = None
    strategy: Optional[str] = None
    language: Optional[str] = None
    round: int = 0
    alive: bool = True
    prompt: Optional[str] = None
    response: Optional[str] = None
    public_neighbors: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phrase": self.phrase,
            "similarity": self.similarity,
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "strategy": self.strategy,
            "language": self.language,
            "round": self.round,
            "alive": self.alive,
            "prompt": self.prompt,
            "response": self.response,
            "public_neighbors": list(self.public_neighbors or []),
        }


@dataclass
class OrchestratorConfig:
    """How the orchestrator fans out calls and prunes live nodes."""

    strategies: List[str] = field(
        default_factory=lambda: list(WHITEBOX_FUZZ_STRATEGIES)
    )
    translate_languages: List[str] = field(
        default_factory=lambda: list(DEFAULT_TRANSLATE_LANGUAGES)
    )
    calls_per_strategy: int = 1
    max_rounds: int = 5
    keep_k: int = 3
    target_similarity: float = 0.95

    def __post_init__(self) -> None:
        self.strategies = normalize_fuzz_strategies(
            self.strategies, fuzz_mode="basic"
        ) or list(WHITEBOX_FUZZ_STRATEGIES)
        langs = [
            lang.strip()
            for lang in self.translate_languages
            if lang and str(lang).strip()
        ]
        self.translate_languages = langs or list(DEFAULT_TRANSLATE_LANGUAGES)
        self.calls_per_strategy = max(1, int(self.calls_per_strategy))
        self.max_rounds = max(1, int(self.max_rounds))
        self.keep_k = max(1, int(self.keep_k))

    @classmethod
    def from_fuzzer_config(cls, cfg: LLMFuzzerConfig) -> "OrchestratorConfig":
        """Build orchestrator settings from ``LLMFuzzerConfig``.

        White-box fields (``whitebox_strategies``, ``calls_per_strategy``,
        ``keep_k``, ``translate_languages``) win. An empty strategy list uses
        the white-box default (paraphrase / translate),
        not explore's ``fuzz_mode`` rotation.
        """
        strategies = list(getattr(cfg, "whitebox_strategies", None) or [])
        languages = list(getattr(cfg, "translate_languages", None) or [])
        return cls(
            strategies=strategies or list(WHITEBOX_FUZZ_STRATEGIES),
            translate_languages=languages or list(DEFAULT_TRANSLATE_LANGUAGES),
            calls_per_strategy=int(getattr(cfg, "calls_per_strategy", 1) or 1),
            max_rounds=int(cfg.max_iterations or 5),
            keep_k=int(getattr(cfg, "keep_k", 3) or 3),
            target_similarity=float(cfg.target_similarity),
        )

    def call_plan(self) -> List[FuzzCall]:
        return build_call_plan(
            self.strategies,
            translate_languages=self.translate_languages,
            calls_per_strategy=self.calls_per_strategy,
        )


def prune_nodes(
    nodes: Sequence[SearchNode],
    keep_k: int,
) -> Tuple[List[SearchNode], List[SearchNode]]:
    """Keep the ``keep_k`` highest-similarity unique phrases; mark the rest dead.

    Ties keep the earlier node (lower round, then node_id). Duplicate phrases
    (case-insensitive) keep the first occurrence after sorting.
    """
    k = max(1, int(keep_k))
    ranked = sorted(
        nodes,
        key=lambda n: (-float(n.similarity), n.round, n.node_id),
    )
    kept: List[SearchNode] = []
    pruned: List[SearchNode] = []
    seen: set[str] = set()
    for node in ranked:
        key = (node.phrase or "").strip().lower()
        if not key or key in seen:
            node.alive = False
            pruned.append(node)
            continue
        if len(kept) < k:
            node.alive = True
            kept.append(node)
            seen.add(key)
        else:
            node.alive = False
            pruned.append(node)
    return kept, pruned


@dataclass
class OrchestratorResult:
    """Outcome of ``FuzzOrchestrator.run``."""

    original_phrase: str
    fuzzed_phrase: str
    final_similarity: float
    baseline_similarity: float
    history: List[str]
    live_nodes: List[SearchNode]
    strategy_rounds: List[Dict[str, Any]]
    interactions: List[Dict[str, Any]]
    call_plan: List[FuzzCall]
    config: OrchestratorConfig
    winning_prompt_chain: List[Dict[str, Any]] = field(default_factory=list)
    live_prompt_chains: List[Dict[str, Any]] = field(default_factory=list)

    def as_fuzz_phrase_tuple(
        self,
    ) -> Tuple[str, float, List[str], List[Dict[str, Any]]]:
        """Shape expected by ``LLMFuzzer.fuzz_phrase`` callers."""
        return (
            self.fuzzed_phrase,
            self.final_similarity,
            self.history,
            self.interactions,
        )


class FuzzOrchestrator:
    """Direct fuzzer calls and maintain the live search-node set."""

    def __init__(
        self,
        fuzzer: LLMFuzzer,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self.fuzzer = fuzzer
        self.config = config or OrchestratorConfig.from_fuzzer_config(fuzzer.config)
        self._seq = 0

    def _next_id(self, round_num: int) -> str:
        self._seq += 1
        return f"r{round_num}_{self._seq}"

    def run(
        self,
        original_phrase: str,
        target_concept: str,
        index: Any,
    ) -> OrchestratorResult:
        """Expand live nodes by the call plan, score, prune, repeat."""
        plan = self.config.call_plan()
        logger.info(
            "Orchestrator start phrase=%r target=%r plan=%s keep_k=%s rounds=%s",
            original_phrase,
            target_concept,
            [c.label for c in plan],
            self.config.keep_k,
            self.config.max_rounds,
        )

        emb_model = self.fuzzer._embedding_model_for_index(index)
        target_vectors = self.fuzzer._embed_texts([target_concept], emb_model)
        if not target_vectors:
            logger.warning("Failed to embed target concept; aborting orchestrator")
            seed = SearchNode(
                phrase=original_phrase,
                similarity=0.0,
                node_id=self._next_id(0),
                round=0,
            )
            seed_chain = prompt_chain_from_lineage([seed])
            return OrchestratorResult(
                original_phrase=original_phrase,
                fuzzed_phrase=original_phrase,
                final_similarity=0.0,
                baseline_similarity=0.0,
                history=[original_phrase],
                live_nodes=[seed],
                strategy_rounds=[],
                interactions=[],
                call_plan=plan,
                config=self.config,
                winning_prompt_chain=seed_chain,
                live_prompt_chains=[
                    {"node_id": seed.node_id, "similarity": 0.0, "chain": seed_chain}
                ],
            )

        target_emb = target_vectors[0]
        baseline_scores = self.fuzzer._similarities_to_target(
            [original_phrase], target_emb, emb_model
        )
        baseline = baseline_scores[0] if baseline_scores else 0.0
        seed = SearchNode(
            phrase=original_phrase,
            similarity=baseline,
            node_id=self._next_id(0),
            round=0,
        )
        live = [seed]
        nodes_by_id: Dict[str, SearchNode] = {seed.node_id: seed}
        elite = seed
        strategy_rounds: List[Dict[str, Any]] = []
        interactions: List[Dict[str, Any]] = []

        for round_idx in range(self.config.max_rounds):
            round_num = round_idx + 1
            live_before = list(live)
            logger.info(
                "Orchestrator round %d/%d live=%d calls_per_node=%d",
                round_num,
                self.config.max_rounds,
                len(live),
                len(plan),
            )
            answers: List[SearchNode] = []
            trials: List[Dict[str, Any]] = []

            for parent in live_before:
                similar = self.fuzzer.find_similar_phrases(parent.phrase, index) or []
                public_neighbors = compact_public_neighbors(similar)
                for call in plan:
                    raw, prompt, response = self.fuzzer.modify_text(
                        parent.phrase,
                        call.strategy,
                        language=call.language,
                        target_concept=target_concept,
                        similar_phrases=similar,
                        repeat_index=call.repeat_index,
                        index=index,
                    )
                    phrase = (
                        normalize_text(raw, self.fuzzer.normalization_config)
                        if raw
                        else ""
                    )
                    score = -1.0
                    node: Optional[SearchNode] = None
                    if phrase:
                        scored = self.fuzzer._similarities_to_target(
                            [phrase], target_emb, emb_model
                        )
                        score = scored[0] if scored else 0.0
                        node = SearchNode(
                            phrase=phrase,
                            similarity=score,
                            node_id=self._next_id(round_num),
                            parent_id=parent.node_id,
                            strategy=call.strategy,
                            language=call.language,
                            round=round_num,
                            alive=False,
                            prompt=prompt,
                            response=response,
                            public_neighbors=public_neighbors,
                        )
                        answers.append(node)
                        nodes_by_id[node.node_id] = node
                    trial = {
                        "round": round_num,
                        "strategy": call.strategy,
                        "language": call.language,
                        "repeat_index": call.repeat_index,
                        "parent_phrase": parent.phrase,
                        "parent_id": parent.node_id,
                        "prompt": prompt,
                        "response": response,
                        "phrase": phrase,
                        "similarity": score,
                        "kept": False,
                        "pruned": False,
                        "node_id": node.node_id if node else None,
                        "public_neighbors": public_neighbors,
                    }
                    trials.append(trial)
                    interactions.append(trial)
                    logger.info(
                        "Round %d [%s] parent=%s similarity=%.3f",
                        round_num,
                        call.label,
                        parent.node_id,
                        score,
                    )

            pool = list(live_before) + answers
            kept, pruned = prune_nodes(pool, self.config.keep_k)
            live = kept
            kept_ids = {n.node_id for n in kept}
            for trial in trials:
                nid = trial.get("node_id")
                if nid and nid in kept_ids:
                    trial["kept"] = True
                else:
                    trial["pruned"] = True

            round_best = max(pool, key=lambda n: n.similarity)
            if round_best.similarity > elite.similarity:
                elite = round_best

            strategy_rounds.append(
                {
                    "round": round_num,
                    "live_before": [n.as_dict() for n in live_before],
                    "trials": [
                        {
                            "strategy": t["strategy"],
                            "language": t.get("language"),
                            "phrase": t["phrase"],
                            "similarity": t["similarity"],
                            "kept": t["kept"],
                            "pruned": t["pruned"],
                            "parent_phrase": t["parent_phrase"],
                        }
                        for t in trials
                    ],
                    "live_after": [n.as_dict() for n in live],
                    "pruned": [n.as_dict() for n in pruned],
                    "kept_strategy": round_best.strategy,
                    "moved": elite.round == round_num,
                    "current_phrase": elite.phrase,
                    "current_similarity": elite.similarity,
                    "active_after": [c.label for c in plan],
                }
            )

            if elite.similarity >= self.config.target_similarity:
                logger.info("Target similarity reached: %.3f", elite.similarity)
                break

        winning_lineage = walk_lineage(elite, nodes_by_id)
        winning_chain = prompt_chain_from_lineage(winning_lineage)
        history = [n.phrase for n in winning_lineage] or [original_phrase]
        live_chains = [
            {
                "node_id": node.node_id,
                "similarity": node.similarity,
                "phrase": node.phrase,
                "chain": prompt_chain_from_lineage(walk_lineage(node, nodes_by_id)),
            }
            for node in live
        ]

        interactions.append(
            {
                "tournament": True,
                "orchestrator": True,
                "strategy_rounds": strategy_rounds,
                "pruned_strategies": [],
                "surviving_strategies": list(self.config.strategies),
                "baseline_similarity": baseline,
                "live_nodes": [n.as_dict() for n in live],
                "keep_k": self.config.keep_k,
                "calls_per_strategy": self.config.calls_per_strategy,
                "call_plan": [
                    {
                        "strategy": c.strategy,
                        "language": c.language,
                        "repeat_index": c.repeat_index,
                    }
                    for c in plan
                ],
                "translate_languages": list(self.config.translate_languages),
                "winning_prompt_chain": winning_chain,
                "live_prompt_chains": live_chains,
            }
        )

        return OrchestratorResult(
            original_phrase=original_phrase,
            fuzzed_phrase=elite.phrase,
            final_similarity=elite.similarity,
            baseline_similarity=baseline,
            history=history,
            live_nodes=live,
            strategy_rounds=strategy_rounds,
            interactions=interactions,
            call_plan=plan,
            config=self.config,
            winning_prompt_chain=winning_chain,
            live_prompt_chains=live_chains,
        )
