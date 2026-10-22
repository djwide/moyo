"""Same-fact collapse of a map scan against claims already on the map.

One grouping covers new claims and prior map nodes. Survivors stay in the
claim list. Matches against an existing node are dropped and recorded as
source-model updates for ingest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pipeline.cluster import (
    _claim_models,
    group_claim_ids_with_ollama,
    merge_claim_group,
)


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def first_seen_models(groups: list[list[str]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for model in group:
            trimmed = str(model or "").strip()
            if not trimmed or trimmed in seen:
                continue
            seen.add(trimmed)
            out.append(trimmed)
    return out


@dataclass
class _Slot:
    slot_id: str
    claim: str
    norm: str
    source_models: list[str]
    kind: str
    receive: bool
    node_id: str | None = None
    members: list[dict] = field(default_factory=list)
    original_models: list[str] = field(default_factory=list)


def _anchor_slots(anchors: list[dict]) -> list[_Slot]:
    slots: list[_Slot] = []
    for index, anchor in enumerate(anchors or []):
        if not isinstance(anchor, dict):
            continue
        node_id = str(anchor.get("nodeId") or anchor.get("node_id") or "").strip()
        claim = str(anchor.get("claim") or "").strip()
        norm = _norm(claim)
        if not node_id or not norm:
            continue
        models = anchor.get("sourceModels")
        if not isinstance(models, list):
            models = anchor.get("source_models") if isinstance(anchor.get("source_models"), list) else []
        source_models = first_seen_models([[str(model) for model in models]])
        section = bool(anchor.get("section")) or node_id.startswith("section_")
        receive = node_id.startswith("claim_") and not section
        slots.append(
            _Slot(
                slot_id=f"A{index}",
                claim=claim,
                norm=norm,
                source_models=list(source_models),
                kind="anchor",
                receive=receive,
                node_id=node_id,
                original_models=list(source_models),
            )
        )
    return slots


def _apply_groups(slots: list[_Slot], groups: list[list[str]]) -> None:
    by_id = {slot.slot_id: slot for slot in slots}
    absorbed: set[str] = set()
    for group in groups:
        members = [by_id[slot_id] for slot_id in group if slot_id in by_id and slot_id not in absorbed]
        news = [slot for slot in members if slot.kind == "new"]
        if not news:
            continue
        receiving = [slot for slot in members if slot.kind == "anchor" and slot.receive]
        blocking = [slot for slot in members if slot.kind == "anchor" and not slot.receive]
        if not receiving and blocking:
            for claim in news:
                absorbed.add(claim.slot_id)
            continue
        keeper = receiving[0] if receiving else news[0]
        for claim in news:
            if claim is keeper:
                continue
            if keeper.receive or keeper.kind == "new":
                keeper.source_models = first_seen_models([keeper.source_models, claim.source_models])
            if keeper.kind == "new":
                keeper.members.extend(claim.members)
            absorbed.add(claim.slot_id)
    if absorbed:
        slots[:] = [slot for slot in slots if slot.slot_id not in absorbed]


def cluster_against_map(
    claims: list[dict],
    anchors: list[dict] | None = None,
    *,
    corroboration_min_sources: int = 2,
    llm_config: dict | None = None,
    prompt_path: Path | str | None = None,
    dry_run: bool = False,
    group_fn: Callable[[list[dict]], list[list[str]]] | None = None,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Collapse this scan against the live map.

    Returns survivors, cluster rows for scoring, and ``map_collapse`` for ingest.
    Exact normalized text merges before the utility model. Word overlap does not.
    """
    indexed = [claim for claim in claims or [] if str(claim.get("claim_id") or "").strip()]
    slots = _anchor_slots(list(anchors or []))
    pending: list[dict] = []
    for claim in indexed:
        text = str(claim.get("claim") or "").strip()
        norm = _norm(text)
        if not norm:
            continue
        models = _claim_models(claim)
        match = next((slot for slot in slots if slot.norm == norm), None)
        if match is not None:
            if match.receive or match.kind == "new":
                match.source_models = first_seen_models([match.source_models, models])
            if match.kind == "new":
                match.members.append(claim)
            continue
        slots.append(
            _Slot(
                slot_id=f"N{len(pending)}",
                claim=text,
                norm=norm,
                source_models=list(models),
                kind="new",
                receive=True,
                members=[claim],
            )
        )
        pending.append(claim)

    has_new = any(slot.kind == "new" for slot in slots)
    if has_new and len(slots) > 1 and not dry_run:
        pseudo = [
            {"claim_id": slot.slot_id, "claim": slot.claim, "source_model": slot.kind}
            for slot in slots
        ]
        if group_fn is not None:
            groups = group_fn(pseudo)
        else:
            cfg = dict(llm_config or {})
            groups = group_claim_ids_with_ollama(
                pseudo,
                client=_cluster_client(cfg),
                prompt_path=Path(prompt_path) if prompt_path else None,
                batch_size=int(cfg.get("batch_size", 35)),
            )
        _apply_groups(slots, groups)

    survivors: list[dict] = []
    clusters: list[dict] = []
    for index, slot in enumerate(slot for slot in slots if slot.kind == "new"):
        merged = merge_claim_group(
            slot.members,
            cluster_id=f"CL{index + 1:03d}",
            corroboration_min_sources=corroboration_min_sources,
        )
        merged["source_models"] = list(slot.source_models)
        merged["source_model"] = slot.source_models[0] if slot.source_models else merged.get("source_model")
        merged["corroboration"] = max(1, len(slot.source_models))
        survivors.append(merged)
        clusters.append(
            {
                "cluster_id": merged["cluster_id"],
                "claim_ids": [merged["claim_id"]],
                "member_ids": list(merged.get("merged_from") or [merged["claim_id"]]),
                "models": list(slot.source_models),
                "citations": list(merged.get("citations") or []),
                "representative_id": merged["claim_id"],
                "size": int(merged.get("merged_count") or 1),
            }
        )

    anchor_updates = []
    for slot in slots:
        if slot.kind != "anchor" or not slot.receive or not slot.node_id:
            continue
        if slot.source_models == slot.original_models:
            continue
        anchor_updates.append(
            {
                "id": slot.node_id,
                "source_models": list(slot.source_models),
                "corroboration": max(len(slot.source_models), len(slot.original_models)),
            }
        )

    new_slots = sum(1 for slot in slots if slot.kind == "new")
    started = sum(1 for claim in indexed if _norm(str(claim.get("claim") or "")))
    collapse = {
        "collapsed": True,
        "duplicate_count": max(0, started - new_slots),
        "anchor_updates": anchor_updates,
    }
    return survivors, clusters, collapse


def _cluster_client(cfg: dict) -> Any:
    from pipeline.cluster import _make_ollama_client

    return _make_ollama_client(cfg)
