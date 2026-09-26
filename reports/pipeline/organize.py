"""Group scored claims into document sections, and optionally suggest review labels."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPORTS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORGANIZE_PROMPT = REPORTS_ROOT / "prompts" / "organize_claims.md"
DEFAULT_LABEL_PROMPT = REPORTS_ROOT / "prompts" / "label_claims.md"
REVIEW_LABELS = ("known", "known_to_be_wrong", "investigate", "not_relevant")
BATCH_SIZE = 40


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
        claim = " ".join(str(row.get("claim") or "").split())
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
            text = client.complete(_render_prompt(DEFAULT_ORGANIZE_PROMPT, batch))
            parsed = parse_sections(text or "", {row["claim_id"] for row in batch})
            if parsed is None:
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


def suggest_review_labels(claims: list[dict[str, str]], *, client: Any) -> dict[str, str]:
    """Return review labels. A failed batch leaves those claims unlabeled."""
    if not claims:
        return {}
    labels: dict[str, str] = {}
    for start in range(0, len(claims), BATCH_SIZE):
        batch = claims[start : start + BATCH_SIZE]
        try:
            text = client.complete(_render_prompt(DEFAULT_LABEL_PROMPT, batch))
        except Exception as exc:
            logger.warning("review label suggestion failed: %s", exc)
            continue
        parsed = parse_review_labels(text or "", {row["claim_id"] for row in batch})
        if not parsed:
            continue
        labels.update(parsed)
    return {claim_id: label for claim_id, label in labels.items() if label in REVIEW_LABELS}


def _write_labels(report: dict[str, Any], labels: dict[str, str]) -> None:
    if not labels:
        return
    for key in ("findings_all", "findings"):
        rows = report.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = labels.get(str(row.get("claim_id") or "").strip())
            if label:
                row["customerLabels"] = [label]


def apply_document_graph(run_dir: Path, *, auto_label: bool, client: Any | None = None) -> None:
    """Write sections onto report_data.json. Failures leave claims in place."""
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
    active = client
    if claims and active is None and (len(claims) >= 2 or auto_label):
        try:
            from moyo.llm.utility import get_utility_llm

            active = get_utility_llm()
        except Exception as exc:
            logger.warning("utility model unavailable for grouping: %s", exc)
            active = None
    report["sections"] = (
        organize_claims(claims, client=active) if active is not None and len(claims) >= 2 else []
    )
    if auto_label and active is not None and claims:
        _write_labels(report, suggest_review_labels(claims, client=active))
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
