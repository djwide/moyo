"""Build a tree_layout from scored claims. Does not rewrite report_data.json."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPORTS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORGANIZE_PROMPT = REPORTS_ROOT / "prompts" / "organize_claims.md"
DEFAULT_TREE_PROMPT = REPORTS_ROOT / "prompts" / "tree_claims.md"
DEFAULT_PLACE_PROMPT = REPORTS_ROOT / "prompts" / "place_find_more.md"
DEFAULT_LABEL_PROMPT = REPORTS_ROOT / "prompts" / "label_claims.md"
REVIEW_LABELS = ("known", "known_to_be_wrong", "investigate", "not_relevant")
BATCH_SIZE = 40
# Gemini Flash spends most of a short completion budget on reasoning tokens;
# 800–1024 caps truncate mid-JSON (finish_reason=length) and wipe sections.
ORGANIZE_MAX_TOKENS = 4096


def finding_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The same list ingest prefers: findings_all, otherwise findings."""
    rows = report.get("findings_all")
    if not isinstance(rows, list):
        rows = report.get("findings")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _claim_payload(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        claim_id = str(row.get("claim_id") or "").strip()
        claim = " ".join(str(row.get("claim") or row.get("text") or "").split())
        if not claim_id or not claim or claim_id in seen:
            continue
        seen.add(claim_id)
        payload.append({"claim_id": claim_id, "claim": claim[:500]})
    return payload


def _load_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_sections(text: str, known_ids: set[str]) -> list[dict[str, Any]] | None:
    """Accept confident sections. None means the grouping must be discarded."""
    data = _load_json_object(text)
    if data is None or not isinstance(data.get("sections"), list):
        return None
    sections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in data["sections"]:
        if not isinstance(item, dict):
            return None
        title = " ".join(str(item.get("title") or "").split())[:80]
        raw_ids = item.get("claim_ids")
        if raw_ids is None:
            raw_ids = item.get("claimIds")
        if not title or not isinstance(raw_ids, list):
            return None
        claim_ids: list[str] = []
        for raw_id in raw_ids:
            claim_id = str(raw_id or "").strip()
            if not claim_id:
                return None
            if claim_id not in known_ids:
                continue
            if claim_id in seen or claim_id in claim_ids:
                return None
            claim_ids.append(claim_id)
        if len(claim_ids) < 2:
            continue
        seen.update(claim_ids)
        sections.append({"title": title, "claim_ids": claim_ids})
    return sections


def _has_cycle(parents: dict[str, str]) -> bool:
    """True when following parent links loops."""
    for start in parents:
        seen: set[str] = set()
        current = start
        while current in parents:
            if current in seen:
                return True
            seen.add(current)
            current = parents[current]
    return False


def parse_claim_parents(text: str, known_ids: set[str]) -> dict[str, str] | None:
    """Map child claim id → parent claim id. None means the payload was unusable."""
    data = _load_json_object(text)
    if data is None or not isinstance(data.get("parents"), list):
        return None
    parents: dict[str, str] = {}
    for item in data["parents"]:
        if not isinstance(item, dict):
            return None
        child = str(item.get("claim_id") or item.get("claimId") or "").strip()
        parent = str(
            item.get("parent_claim_id") or item.get("parentClaimId") or item.get("parent_id") or ""
        ).strip()
        if not child or not parent:
            return None
        if child not in known_ids or parent not in known_ids:
            continue
        if child == parent:
            return None
        if child in parents and parents[child] != parent:
            return None
        parents[child] = parent
    if _has_cycle(parents):
        return None
    return parents


def parse_review_labels(text: str, known_ids: set[str]) -> dict[str, str] | None:
    """Map claim id to one review label. None means the payload was unusable."""
    data = _load_json_object(text)
    if data is None or not isinstance(data.get("labels"), list):
        return None
    labels: dict[str, str] = {}
    for item in data["labels"]:
        if not isinstance(item, dict):
            continue
        claim_id = str(item.get("claim_id") or item.get("claimId") or "").strip()
        label = str(item.get("label") or "").strip().lower()
        if claim_id not in known_ids or claim_id in labels:
            continue
        if label not in REVIEW_LABELS:
            continue
        labels[claim_id] = label
    return labels


def _render_prompt(path: Path, claims: list[dict[str, str]]) -> str:
    template = path.read_text(encoding="utf-8")
    payload = json.dumps(claims, ensure_ascii=False)
    return template.replace("{{ claims_json }}", payload)


def _render_place_prompt(
    new_claims: list[dict[str, str]], existing_nodes: list[dict[str, Any]]
) -> str:
    template = DEFAULT_PLACE_PROMPT.read_text(encoding="utf-8")
    existing_payload = [
        {
            "claim_id": str(row.get("claim_id") or "").strip(),
            "claim": str(row.get("claim") or ""),
            "section": bool(row.get("section")),
        }
        for row in existing_nodes
        if str(row.get("claim_id") or "").strip()
    ]
    return template.replace(
        "{{ existing_json }}", json.dumps(existing_payload, ensure_ascii=False)
    ).replace("{{ claims_json }}", json.dumps(new_claims, ensure_ascii=False))


def anchor_nodes_from_prior_claims(prior: Any) -> list[dict[str, Any]]:
    """Compact existing graph nodes for attach-mode placement."""
    rows = prior if isinstance(prior, list) else []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("nodeId") or item.get("claim_id") or "").strip()
        claim = " ".join(str(item.get("claim") or "").split())
        if not node_id or not claim or node_id in seen:
            continue
        seen.add(node_id)
        depth_raw = item.get("depth")
        try:
            depth = int(depth_raw) if depth_raw is not None else 1
        except (TypeError, ValueError):
            depth = 1
        section = bool(item.get("section")) or node_id.startswith("section_")
        out.append(
            {
                "claim_id": node_id,
                "claim": claim[:500],
                "section": section,
                "depth": max(1, depth),
            }
        )
    return out


def parse_attach_parents(
    text: str, new_ids: set[str], known_ids: set[str]
) -> dict[str, str] | None:
    """Child→parent for new claims only. None means the payload was unusable."""
    data = _load_json_object(text)
    if data is None or not isinstance(data.get("parents"), list):
        return None
    parents: dict[str, str] = {}
    for item in data["parents"]:
        if not isinstance(item, dict):
            return None
        child = str(item.get("claim_id") or item.get("claimId") or "").strip()
        parent = str(
            item.get("parent_claim_id") or item.get("parentClaimId") or item.get("parent_id") or ""
        ).strip()
        if not child or not parent:
            return None
        if child not in new_ids:
            continue
        if parent not in known_ids:
            continue
        if child == parent:
            return None
        if child in parents and parents[child] != parent:
            return None
        parents[child] = parent
    if _has_cycle(parents):
        return None
    return parents


def organize_claims(claims: list[dict[str, str]], *, client: Any) -> list[dict[str, Any]]:
    """Return sections, or [] when the model output cannot be trusted."""
    if len(claims) < 2:
        return []
    known_ids = {row["claim_id"] for row in claims}
    sections: list[dict[str, Any]] = []
    try:
        for start in range(0, len(claims), BATCH_SIZE):
            batch = claims[start : start + BATCH_SIZE]
            if len(batch) < 2:
                continue
            text = client.complete(
                _render_prompt(DEFAULT_ORGANIZE_PROMPT, batch),
                max_tokens=ORGANIZE_MAX_TOKENS,
            )
            parsed = parse_sections(text or "", {row["claim_id"] for row in batch})
            if parsed is None:
                logger.warning(
                    "claim grouping discarded batch at offset %s (%s claims, %s chars); "
                    "likely truncated or invalid JSON",
                    start,
                    len(batch),
                    len(text or ""),
                )
                return []
            sections.extend(parsed)
    except Exception as exc:
        logger.warning("claim grouping failed: %s", exc)
        return []
    overlap: set[str] = set()
    kept: list[dict[str, Any]] = []
    for section in sections:
        ids = [claim_id for claim_id in section["claim_ids"] if claim_id in known_ids]
        if any(claim_id in overlap for claim_id in ids):
            return []
        if len(ids) < 2:
            continue
        overlap.update(ids)
        kept.append({"title": section["title"], "claim_ids": ids})
    return kept


def tree_claims(claims: list[dict[str, str]], *, client: Any) -> dict[str, str]:
    """Return child→parent claim ids, or {} when the model output cannot be trusted."""
    if len(claims) < 2:
        return {}
    known_ids = {row["claim_id"] for row in claims}
    parents: dict[str, str] = {}
    try:
        for start in range(0, len(claims), BATCH_SIZE):
            batch = claims[start : start + BATCH_SIZE]
            if len(batch) < 2:
                continue
            text = client.complete(
                _render_prompt(DEFAULT_TREE_PROMPT, batch),
                max_tokens=ORGANIZE_MAX_TOKENS,
            )
            parsed = parse_claim_parents(text or "", {row["claim_id"] for row in batch})
            if parsed is None:
                logger.warning(
                    "claim tree discarded batch at offset %s (%s claims, %s chars)",
                    start,
                    len(batch),
                    len(text or ""),
                )
                # Discard this batch only; keep earlier confident links.
                continue
            for child, parent in parsed.items():
                if child in parents and parents[child] != parent:
                    continue
                if child == parent or child not in known_ids or parent not in known_ids:
                    continue
                parents[child] = parent
                if _has_cycle(parents):
                    del parents[child]
    except Exception as exc:
        logger.warning("claim tree failed: %s", exc)
        return {}
    if _has_cycle(parents):
        return {}
    return parents


def place_find_more_claims(
    new_claims: list[dict[str, str]],
    existing_nodes: list[dict[str, Any]],
    *,
    client: Any,
) -> dict[str, str]:
    """Attach new claims to existing nodes or other new claims. {} if unusable."""
    if not new_claims:
        return {}
    new_ids = {row["claim_id"] for row in new_claims}
    known_ids = new_ids | {
        str(row.get("claim_id") or "").strip()
        for row in existing_nodes
        if str(row.get("claim_id") or "").strip()
    }
    if len(known_ids) < 2:
        return {}
    try:
        text = client.complete(
            _render_place_prompt(new_claims, existing_nodes),
            max_tokens=ORGANIZE_MAX_TOKENS,
        )
    except Exception as exc:
        logger.warning("find-more claim placement failed: %s", exc)
        return {}
    parsed = parse_attach_parents(text or "", new_ids, known_ids)
    if parsed is None:
        logger.warning("find-more claim placement discarded unusable JSON")
        return {}
    parents = dict(parsed)
    if _has_cycle(parents):
        return {}
    return parents


def suggest_review_labels(claims: list[dict[str, str]], *, client: Any) -> dict[str, str]:
    """Return review labels. A failed batch leaves those claims unlabeled."""
    if not claims:
        return {}
    labels: dict[str, str] = {}
    for start in range(0, len(claims), BATCH_SIZE):
        batch = claims[start : start + BATCH_SIZE]
        try:
            text = client.complete(
                _render_prompt(DEFAULT_LABEL_PROMPT, batch),
                max_tokens=ORGANIZE_MAX_TOKENS,
            )
        except Exception as exc:
            logger.warning("review label suggestion failed: %s", exc)
            continue
        parsed = parse_review_labels(text or "", {row["claim_id"] for row in batch})
        if not parsed:
            logger.warning(
                "review labels empty for batch at offset %s (%s claims, %s chars)",
                start,
                len(batch),
                len(text or ""),
            )
            continue
        labels.update(parsed)
    return {claim_id: label for claim_id, label in labels.items() if label in REVIEW_LABELS}


ROOT_ID = "root"


def build_tree_layout(
    *,
    topic: str,
    claims: list[dict[str, str]],
    sections: list[dict[str, Any]],
    parents: dict[str, str],
    labels: dict[str, str],
    existing_depths: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build tree_layout.json. Parent and depth live on each node; no edges.

    Layer-1 nodes (sections and ungrouped claims) use parent=\"root\". Claim text
    stays on report_data.json. Existing graph ids may appear only as parents.
    """
    nested = {child for child, parent in parents.items() if child and parent and child != parent}
    section_of: dict[str, str] = {}
    nodes: list[dict[str, Any]] = []
    new_claim_ids = {
        str(row.get("claim_id") or "").strip()
        for row in claims
        if str(row.get("claim_id") or "").strip()
    }
    for index, section in enumerate(sections):
        section_id = f"section:{index}"
        title = str(section.get("title") or "").strip()
        claim_ids = [
            str(claim_id).strip()
            for claim_id in (section.get("claim_ids") or [])
            if str(claim_id).strip()
        ]
        if not title:
            continue
        nodes.append(
            {
                "id": section_id,
                "kind": "section",
                "title": title,
                "parent": ROOT_ID,
                "depth": 1,
            }
        )
        for claim_id in claim_ids:
            if claim_id in nested:
                continue
            section_of[claim_id] = section_id

    section_ids = {str(node["id"]) for node in nodes}
    claim_parent: dict[str, str] = {}
    for row in claims:
        claim_id = str(row.get("claim_id") or "").strip()
        if not claim_id:
            continue
        parent = parents.get(claim_id) or ""
        if parent and parent != claim_id:
            claim_parent[claim_id] = parent
        elif claim_id in section_of:
            claim_parent[claim_id] = section_of[claim_id]
        else:
            claim_parent[claim_id] = ROOT_ID

    depth_of: dict[str, int] = {ROOT_ID: 0}
    for node in nodes:
        depth_of[str(node["id"])] = 1
    if existing_depths:
        for node_id, depth in existing_depths.items():
            key = str(node_id).strip()
            if not key or key in depth_of:
                continue
            try:
                depth_of[key] = max(1, int(depth))
            except (TypeError, ValueError):
                depth_of[key] = 1
    for parent in claim_parent.values():
        if parent in depth_of or parent == ROOT_ID:
            continue
        if parent in new_claim_ids or parent in section_ids:
            continue
        depth_of[parent] = 1

    pending = set(claim_parent)
    while pending:
        progressed = False
        for claim_id in list(pending):
            parent = claim_parent[claim_id]
            if parent not in depth_of:
                continue
            depth_of[claim_id] = depth_of[parent] + 1
            pending.remove(claim_id)
            progressed = True
        if progressed:
            continue
        for claim_id in pending:
            claim_parent[claim_id] = ROOT_ID
            depth_of[claim_id] = 1
        break

    for row in claims:
        claim_id = str(row.get("claim_id") or "").strip()
        if not claim_id:
            continue
        nodes.append(
            {
                "id": claim_id,
                "kind": "claim",
                "parent": claim_parent.get(claim_id, ROOT_ID),
                "depth": depth_of.get(claim_id, 1),
            }
        )

    return {
        "version": 1,
        "topic": topic,
        "root": ROOT_ID,
        "nodes": nodes,
        "review_labels": labels,
    }


def utility_organize_client() -> Any | None:
    """Hosted/local utility model used for sections, claim trees, and label hints."""
    try:
        from moyo.llm.utility import get_utility_llm

        return get_utility_llm()
    except Exception as exc:
        logger.warning("utility model unavailable for claim organization: %s", exc)
        return None


def apply_document_graph(
    run_dir: Path,
    *,
    auto_label: bool,
    client: Any | None = None,
    require_utility: bool = False,
    existing_nodes: list[dict[str, Any]] | None = None,
    attach: bool = False,
    stub_only: bool = False,
) -> None:
    """Write tree_layout.json from scored claims. Leaves report_data.json unchanged."""
    path = run_dir / "report_data.json"
    if not path.exists():
        return
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read report_data.json: %s", exc)
        return
    if not isinstance(report, dict):
        return
    claims = _claim_payload(finding_rows(report))
    anchors = existing_nodes or []
    existing_depths: dict[str, int] = {}
    for row in anchors:
        key = str(row.get("claim_id") or "").strip()
        if not key:
            continue
        try:
            existing_depths[key] = max(1, int(row.get("depth") or 1))
        except (TypeError, ValueError):
            existing_depths[key] = 1
    if stub_only:
        layout = build_tree_layout(
            topic=str(report.get("topic") or ""),
            claims=claims,
            sections=[],
            parents={},
            labels={},
        )
        (run_dir / "tree_layout.json").write_text(
            json.dumps(layout, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return
    utility = client or utility_organize_client()
    needs_utility = len(claims) >= 2 or (attach and claims and anchors)
    if require_utility and needs_utility and utility is None:
        raise RuntimeError("Utility LLM is required to build the claim-parent tree.")
    if attach and utility is not None and claims and anchors:
        parents = place_find_more_claims(claims, anchors, client=utility)
        nested = set(parents)
        to_group = [row for row in claims if row["claim_id"] not in nested]
        sections = (
            organize_claims(to_group, client=utility) if len(to_group) >= 2 else []
        )
    else:
        sections = (
            organize_claims(claims, client=utility)
            if utility is not None and len(claims) >= 2
            else []
        )
        parents = (
            tree_claims(claims, client=utility)
            if utility is not None and len(claims) >= 2
            else {}
        )
        existing_depths = {}
    labels = (
        suggest_review_labels(claims, client=utility)
        if auto_label and utility is not None and claims
        else {}
    )
    layout = build_tree_layout(
        topic=str(report.get("topic") or ""),
        claims=claims,
        sections=sections,
        parents=parents,
        labels=labels,
        existing_depths=existing_depths or None,
    )
    (run_dir / "tree_layout.json").write_text(
        json.dumps(layout, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
