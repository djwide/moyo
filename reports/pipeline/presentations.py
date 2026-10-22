"""Display-language sidecar for a map scan. Canonical claims stay English."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

BATCH = 12

_ENGLISH = {"", "english", "en", "eng"}


def _english(language: str) -> bool:
    return (language or "").strip().lower() in _ENGLISH


def _load_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
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


def _presentation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    out: dict[str, Any] = {}
    claim = value.get("claim")
    title = value.get("title")
    summary = value.get("summary")
    if isinstance(claim, str) and claim.strip():
        out["claim"] = claim.strip()[:8000]
    if isinstance(title, str) and title.strip():
        out["title"] = title.strip()[:120]
    if isinstance(summary, str) and summary.strip():
        out["summary"] = " ".join(summary.split())[:120]
    return out or None


def _prompt(language: str, rows: dict[str, dict[str, Any]]) -> str:
    return "\n\n".join(
        [
            f"Translate the JSON values into {language}.",
            "Return only a JSON object with the same ids.",
            "Each value is an object with claim, title, and summary.",
            "Keep proper names, numbers, and URLs unchanged. Do not add facts.",
            "Use null for title or summary when the source value is null.",
            "If a string is already in the target language, return it unchanged.",
            json.dumps(rows, ensure_ascii=False),
        ]
    )


def translate_presentations(
    language: str,
    rows: dict[str, dict[str, Any]],
    *,
    complete: Callable[[str], str],
) -> dict[str, dict[str, Any]]:
    """Translate id -> {claim, title, summary}. Empty when the language is English."""
    if _english(language) or not rows:
        return {}
    translated: dict[str, dict[str, Any]] = {}
    keys = list(rows)
    for index in range(0, len(keys), BATCH):
        chunk = {key: rows[key] for key in keys[index : index + BATCH]}
        parsed = _load_object(complete(_prompt(language, chunk)))
        if not parsed:
            continue
        for key in chunk:
            presentation = _presentation(parsed.get(key))
            if presentation:
                translated[key] = presentation
    return translated


def _section_key(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def write_map_presentations(
    run_dir: Path,
    language: str,
    *,
    complete: Callable[[str], str] | None = None,
) -> Path | None:
    """Write presentations.json beside report_data. English and failures leave no file."""
    if _english(language):
        return None
    report_path = run_dir / "report_data.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    findings = report.get("findings_all")
    if not isinstance(findings, list):
        findings = report.get("findings") if isinstance(report.get("findings"), list) else []
    rows: dict[str, dict[str, Any]] = {}
    claim_ids: list[str] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        claim_id = str(finding.get("claim_id") or "").strip()
        claim = str(finding.get("claim") or "").strip()
        if not claim_id or not claim:
            continue
        summary = finding.get("summary")
        rows[claim_id] = {
            "claim": claim,
            "title": None,
            "summary": summary.strip() if isinstance(summary, str) and summary.strip() else None,
        }
        claim_ids.append(claim_id)
    section_keys: dict[str, str] = {}
    layout_path = run_dir / "tree_layout.json"
    if layout_path.is_file():
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        nodes = layout.get("nodes") if isinstance(layout, dict) else None
        if isinstance(nodes, list):
            for node in nodes:
                if not isinstance(node, dict) or node.get("kind") != "section":
                    continue
                title = str(node.get("title") or "").strip()
                key = _section_key(title)
                if not title or not key:
                    continue
                slot = f"section:{key}"
                rows[slot] = {"claim": title, "title": title, "summary": None}
                section_keys[slot] = key
    if not rows:
        return None
    if complete is None:
        from moyo.llm.utility import get_utility_llm

        client = get_utility_llm()
        complete = client.complete
    translated = translate_presentations(language, rows, complete=complete)
    if not translated:
        return None
    payload = {
        "language": language.strip(),
        "claims": {claim_id: translated[claim_id] for claim_id in claim_ids if claim_id in translated},
        "sections": {section_keys[slot]: translated[slot] for slot in section_keys if slot in translated},
    }
    dest = run_dir / "presentations.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest
