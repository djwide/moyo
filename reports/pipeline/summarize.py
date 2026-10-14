"""3–5 word claim summaries for collapsed map pods.

Live runs use the utility LLM. Dry-run and test mode clip the claim text so
offline pipelines still store a summary.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPORTS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PROMPT = REPORTS_ROOT / "prompts" / "summarize_claims.md"
SUMMARY_BATCH = 24
SUMMARY_MAX_TOKENS = 4096

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_SENTENCE_END = re.compile(r"[.!?。！？]")


def normalize_summary(raw: str) -> str | None:
    """Accept a 3–5 word phrase, or a short space-less phrase in CJK scripts."""
    text = " ".join(str(raw or "").replace("\n", " ").split())
    text = text.strip("\"'`“”‘’")
    text = text.rstrip(".,;:!?…").strip()
    if not text or _SENTENCE_END.search(text):
        return None
    words = text.split()
    if 3 <= len(words) <= 5:
        return " ".join(words)
    if len(words) == 1 and _short_spaceless_phrase(text):
        return text
    return None


def _short_spaceless_phrase(text: str) -> bool:
    letters = _CJK.findall(text)
    return 3 <= len(letters) <= 12 and len(text) <= 16


def clip_claim_summary(claim: str) -> str | None:
    """Offline stand-in: the first few words of the claim."""
    words = " ".join(str(claim or "").split()).split()
    if not words:
        return None
    if len(words) == 1 and _short_spaceless_phrase(words[0]):
        return words[0]
    phrase = " ".join(words[:5]).strip("\"'`“”‘’").rstrip(".,;:!?…").strip()
    return phrase or None


def claim_node_needs_summary(node_id: str, data: dict[str, Any]) -> bool:
    """True for a stored map claim that should receive a summary."""
    if str(node_id or "").startswith("section_"):
        return False
    finding = data.get("finding")
    if isinstance(finding, dict) and finding.get("section") is True:
        return False
    if str(data.get("summary") or "").strip():
        return False
    return bool(" ".join(str(data.get("claim") or "").split()))


def parse_summaries(text: str, known_ids: set[str]) -> dict[str, str]:
    """Map claim id → accepted summary. Invalid rows are dropped."""
    data = _load_json_object(text)
    if data is None or not isinstance(data.get("summaries"), list):
        return {}
    out: dict[str, str] = {}
    for item in data["summaries"]:
        if not isinstance(item, dict):
            continue
        claim_id = str(item.get("claim_id") or item.get("claimId") or "").strip()
        if not claim_id or claim_id not in known_ids or claim_id in out:
            continue
        summary = normalize_summary(str(item.get("summary") or ""))
        if summary:
            out[claim_id] = summary
    return out


def attach_claim_summaries(
    claims: list[dict[str, Any]],
    *,
    client: Any | None = None,
    dry_run: bool = False,
) -> int:
    """Set ``summary`` on claims that do not have one. Returns how many were written."""
    pending = [claim for claim in claims if _needs_summary(claim)]
    if not pending:
        return 0
    if dry_run or _in_test_mode():
        return _clip_pending(pending)
    if client is None:
        try:
            from moyo.llm.utility import get_utility_llm

            client = get_utility_llm()
        except Exception as exc:
            logger.warning("utility model unavailable for claim summaries: %s", exc)
            return 0
    written = 0
    missing = list(pending)
    for _attempt in range(2):
        if not missing:
            break
        still: list[dict[str, Any]] = []
        for start in range(0, len(missing), SUMMARY_BATCH):
            batch = missing[start : start + SUMMARY_BATCH]
            try:
                parsed = _complete_summaries(client, batch)
            except Exception as exc:
                logger.warning("claim summary batch failed: %s", exc)
                parsed = {}
            for claim in batch:
                claim_id = str(claim.get("claim_id") or "").strip()
                summary = parsed.get(claim_id)
                if summary:
                    claim["summary"] = summary
                    written += 1
                else:
                    still.append(claim)
        missing = still
    return written


def summarize_backfill_nodes(
    nodes: list[dict[str, Any]],
    *,
    client: Any | None = None,
    dry_run: bool = False,
) -> list[dict[str, str]]:
    """Return ``{project_id, id, summary}`` updates for nodes that gained a summary.

    ``project_id`` may be empty when the caller is not scoping by map.
    """
    claims: list[dict[str, Any]] = []
    owners: dict[str, tuple[str, str]] = {}
    for index, node in enumerate(nodes):
        node_id = str(node.get("id") or "").strip()
        if not claim_node_needs_summary(node_id, node):
            continue
        claim_id = f"N{index:06d}"
        claims.append(
            {
                "claim_id": claim_id,
                "claim": node.get("claim") or "",
                "language": node.get("language") or "",
            }
        )
        owners[claim_id] = (str(node.get("project_id") or ""), node_id)
    attach_claim_summaries(claims, client=client, dry_run=dry_run)
    updates: list[dict[str, str]] = []
    for claim in claims:
        summary = str(claim.get("summary") or "").strip()
        owner = owners.get(str(claim.get("claim_id") or ""))
        if not summary or owner is None:
            continue
        project_id, node_id = owner
        updates.append({"project_id": project_id, "id": node_id, "summary": summary})
    return updates


def _needs_summary(claim: dict[str, Any]) -> bool:
    if str(claim.get("summary") or "").strip():
        return False
    if not str(claim.get("claim_id") or "").strip():
        return False
    return bool(" ".join(str(claim.get("claim") or "").split()))


def _clip_pending(pending: list[dict[str, Any]]) -> int:
    written = 0
    for claim in pending:
        clipped = clip_claim_summary(str(claim.get("claim") or ""))
        if not clipped:
            continue
        claim["summary"] = clipped
        written += 1
    return written


def _in_test_mode() -> bool:
    try:
        from moyo.llm.testing import is_test_mode

        return is_test_mode()
    except Exception:
        return False


def _complete_summaries(client: Any, batch: list[dict[str, Any]]) -> dict[str, str]:
    payload = _claim_payload(batch)
    known = {row["claim_id"] for row in payload}
    if not known:
        return {}
    prompt = DEFAULT_SUMMARY_PROMPT.read_text(encoding="utf-8").replace(
        "{{ claims_json }}",
        json.dumps(payload, ensure_ascii=False),
    )
    text = client.complete(prompt, max_tokens=SUMMARY_MAX_TOKENS)
    return parse_summaries(text or "", known)


def _claim_payload(claims: list[dict[str, Any]]) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    seen: set[str] = set()
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "").strip()
        text = " ".join(str(claim.get("claim") or "").split())
        if not claim_id or not text or claim_id in seen:
            continue
        seen.add(claim_id)
        row = {"claim_id": claim_id, "claim": text[:500]}
        language = str(claim.get("language") or "").strip()
        if language:
            row["language"] = language
        payload.append(row)
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
