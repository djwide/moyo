"""[3c] Optional LLM synthesis for executive summary + follow-ups."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _fill(tmpl: str, **kwargs: str) -> str:
    out = tmpl
    for k, v in kwargs.items():
        out = out.replace("{{ " + k + " }}", v)
    return out


def _strip_md_noise(text: str) -> str:
    """Remove leftover markdown emphasis / fences that leak into PDFs."""
    text = (text or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Drop **bold** / **** leftovers but keep inner words
    text = re.sub(r"\*{1,}", "", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_SECTION_KEYS = (
    "headline",
    "summary",
    "top_finding_blurb",
    "why_it_matters",
    "confidence_label",
    "confidence_rationale",
    "public_sources",
    "inference_chain",
    "defensive_action",
    "exposure_teaser",
    "exposure_chain",
    "what_else",
    "executive_summary",
    "body",
)


def _parse_labeled_sections(text: str) -> dict[str, Any]:
    """Parse legacy ``**field**\\nvalue`` blobs into a dict."""
    cleaned = _strip_md_noise(text)
    pattern = re.compile(
        r"(?im)^\s*("
        + "|".join(re.escape(k) for k in _SECTION_KEYS)
        + r")\s*:?\s*\n(.*?)(?=^\s*(?:"
        + "|".join(re.escape(k) for k in _SECTION_KEYS)
        + r")\s*:?\s*$|\Z)",
        re.S,
    )
    out: dict[str, Any] = {}
    for m in pattern.finditer(cleaned):
        key = m.group(1).strip().lower()
        body = m.group(2).strip()
        if key in {"public_sources", "inference_chain", "exposure_chain", "what_else"}:
            items = []
            for line in body.splitlines():
                s = re.sub(r"^[-*•\d.)\s]+", "", line).strip()
                if s:
                    items.append(s)
            out[key] = items or [body]
        else:
            out[key] = body
    return out


def parse_executive_payload(raw: Any) -> dict[str, Any]:
    """Normalize LLM / legacy executive_summary into structured fields."""
    if isinstance(raw, dict):
        data = dict(raw)
    else:
        text = _strip_md_noise(str(raw or ""))
        data = {}
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    data = parsed
            except json.JSONDecodeError:
                # Truncated JSON — try labeled fallback on stripped text
                data = _parse_labeled_sections(text)
        if not data:
            data = _parse_labeled_sections(text)
        if not data and text:
            data = {"summary": text}

    def _str(key: str, *alts: str) -> str:
        for k in (key, *alts):
            v = data.get(k)
            if v:
                return _strip_md_noise(str(v))
        return ""

    def _list(key: str, limit: int | None = None) -> list[str]:
        v = data.get(key)
        items: list[str] = []
        if isinstance(v, list):
            items = [_strip_md_noise(str(x)) for x in v if str(x).strip()]
        elif isinstance(v, str) and v.strip():
            for line in v.splitlines():
                s = re.sub(r"^[-*•\d.)\s]+", "", line).strip()
                if s:
                    items.append(_strip_md_noise(s))
        if limit is not None:
            items = items[:limit]
        return [x for x in items if x]

    summary = _str("summary", "executive_summary", "body")
    headline = _str("headline")
    # Legacy blobs put the overview under **headline**; keep title fixed.
    if not summary and headline and len(headline) > 48:
        summary = headline
        headline = "What AI Systems Reveal"
    if not summary:
        summary = _str("top_finding_blurb")
    if not headline or len(headline) > 48:
        headline = "What AI Systems Reveal"
    return {
        "headline": headline,
        "summary": summary,
        "top_finding_blurb": _str("top_finding_blurb") or summary,
        "why_it_matters": _str("why_it_matters"),
        "confidence_label": _str("confidence_label"),
        "confidence_rationale": _str("confidence_rationale"),
        "public_sources": _list("public_sources", 3),
        "inference_chain": _list("inference_chain", 5),
        "defensive_action": _str("defensive_action"),
        "exposure_teaser": _str("exposure_teaser"),
        "exposure_chain": _list("exposure_chain", 6),
        "what_else": _list("what_else", 5),
    }


def synthesize(
    report_data: dict[str, Any],
    *,
    prompts_dir: Path,
    config: dict,
    llm_config: dict | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Enrich report_data with executive_summary and followups."""
    findings_preview = json.dumps(
        report_data.get("findings", [])[:25], ensure_ascii=False, indent=2
    )
    exec_path = prompts_dir / Path(
        config.get("executive_prompt", "prompts/executive_summary.md")
    ).name
    follow_path = prompts_dir / Path(
        config.get("followups_prompt", "prompts/recommend_followups.md")
    ).name

    if not report_data.get("executive_summary"):
        top = report_data.get("top_finding", {}).get("text", "")
        counts = report_data.get("counts", {})
        report_data["executive_summary"] = {
            "headline": "What AI Systems Reveal",
            "summary": (
                f"{counts.get('findings', 0)} findings across "
                f"{counts.get('llms_tested', 0)} models; "
                f"{counts.get('high_sensitivity', 0)} high-sensitivity. "
                f"Lead finding: {top}"
            ),
            "top_finding_blurb": top,
        }

    if not report_data.get("followups"):
        report_data["followups"] = [
            {
                "method": "white-box retest",
                "action": (
                    "Re-prompt high-sensitivity claim holders with constrained "
                    "refusal checks"
                ),
                "claim_ids": [
                    c["claim_id"]
                    for c in report_data.get("findings", [])[:5]
                    if c.get("sensitivity", 0) >= 4
                ],
            },
            {
                "method": "remediation analysis",
                "action": (
                    "Map contested / outlier disclosures to policy and "
                    "retrieval filters"
                ),
                "claim_ids": [
                    c["claim_id"]
                    for c in report_data.get("findings", [])
                    if c.get("status") in {"OUTLIER", "CONTESTED", "MODEL-SPECIFIC"}
                ][:8],
            },
        ]

    # Always normalize whatever is already on disk / from a prior run
    parsed = parse_executive_payload(report_data.get("executive_summary"))
    report_data["executive_summary"] = parsed.get("summary") or parsed.get(
        "top_finding_blurb", ""
    )
    report_data["executive_fields"] = parsed
    if parsed.get("exposure_chain"):
        report_data["exposure_chain"] = parsed["exposure_chain"]
    if parsed.get("what_else"):
        report_data["what_else"] = parsed["what_else"]
    if parsed.get("headline"):
        report_data["headline"] = parsed["headline"]

    try:
        from moyo.llm.testing import is_test_mode
        if is_test_mode():
            dry_run = True
    except Exception:
        pass

    if dry_run:
        return report_data

    try:
        from moyo.llm.client import LLMClient, LLMSpec
    except ImportError:
        return report_data

    if not exec_path.exists():
        return report_data

    llm = llm_config or {}
    try:
        spec = LLMSpec.from_dict(
            {
                "provider": llm.get("provider", "custom"),
                "model": llm.get("model", "kimi-k2.6"),
                "base_url": llm.get("base_url", "https://api.moonshot.ai/v1"),
                "api_key": llm.get("api_key", "$MOONSHOT_API_KEY"),
                "temperature": float(llm.get("temperature", 0.3)),
                "max_tokens": int(llm.get("max_tokens", 1200)),
                "timeout": int(llm.get("timeout", 120)),
            }
        )
        if not spec.api_key:
            return report_data
        client = LLMClient(spec)
        if not client.is_available():
            return report_data
        if exec_path.exists():
            prompt = _fill(
                exec_path.read_text(encoding="utf-8"),
                topic=report_data.get("topic", ""),
                findings_json=findings_preview,
                counts_json=json.dumps(report_data.get("counts", {})),
                report_data_json=json.dumps(
                    {
                        "topic": report_data.get("topic"),
                        "counts": report_data.get("counts"),
                        "top_finding": report_data.get("top_finding"),
                        "findings": report_data.get("findings", [])[:25],
                        "chains": report_data.get("chains", [])[:5],
                        "followups": report_data.get("followups", [])[:5],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            text = (client.complete(prompt) or "").strip()
            if text:
                parsed = parse_executive_payload(text)
                report_data["executive_fields"] = parsed
                report_data["executive_summary"] = (
                    parsed.get("summary") or parsed.get("top_finding_blurb") or ""
                )[:2000]
                if parsed.get("exposure_chain"):
                    report_data["exposure_chain"] = parsed["exposure_chain"]
                if parsed.get("what_else"):
                    report_data["what_else"] = parsed["what_else"]
                if parsed.get("headline"):
                    report_data["headline"] = parsed["headline"]
        if follow_path.exists():
            prompt = _fill(
                follow_path.read_text(encoding="utf-8"),
                findings_json=findings_preview,
            )
            text = (client.complete(prompt) or "").strip()
            arr = _try_json_array(text)
            if arr:
                report_data["followups"] = arr
    except Exception:
        pass
    return report_data


def _try_json_array(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        return []
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []
