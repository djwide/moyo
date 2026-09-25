"""Labels and display grouping for model-, language-, and source-specific claims."""

from __future__ import annotations

import re
from typing import Any

from .audience import OPPOSITION, disclosure_class, normalize_audience

_URL_RE = re.compile(r"https?://", re.IGNORECASE)
_LANG_SUFFIX_RE = re.compile(r"\s*\(([^)]+)\)\s*$")
_MISCONDUCT_RE = re.compile(
    r"harass|sexual|inappropriat|unwanted (?:touch|sexual|comment)|"
    r"hostile work|#?\bmetoo\b|"
    r"denied the (?:sexual |harassment )?alleg|"
    r"false and defamatory|"
    r"office of congressional ethics|\boce\b|"
    r"house (?:committee on )?ethics|"
    r"investigative subcommittee|"
    r"no formal (?:finding|charge|sanction|disciplinary)",
    re.IGNORECASE,
)
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "his", "her", "their",
    "was", "were", "are", "been", "being", "who", "which", "said", "says",
    "according", "report", "reported", "reports", "reporting", "including",
    "included", "junior", "vicente", "gonzalez", "gonzález", "gonzales",
}


def has_source_url(finding: dict[str, Any]) -> bool:
    """True when a citation carries an http(s) URL, not only a named outlet."""
    for raw in finding.get("citations") or []:
        if _URL_RE.search(str(raw or "")):
            return True
    for entry in finding.get("citations_display") or []:
        if isinstance(entry, dict) and _URL_RE.search(str(entry.get("url") or "")):
            return True
    return False


def _model_names(finding: dict[str, Any]) -> list[str]:
    models = finding.get("source_models")
    if isinstance(models, list) and models:
        return [str(m).strip() for m in models if str(m).strip()]
    one = str(finding.get("source_model") or "").strip()
    return [one] if one else []


def _short_model(name: str) -> str:
    text = name.strip()
    while True:
        nxt = _LANG_SUFFIX_RE.sub("", text).strip()
        if nxt == text:
            break
        text = nxt
    return text or name.strip() or "unknown"


def language_specific(finding: dict[str, Any]) -> str:
    """Language name when every attesting model answered in that foreign language."""
    from .language import is_foreign_language

    models = _model_names(finding)
    if not models:
        return ""
    foreign: list[str] = []
    for name in models:
        match = _LANG_SUFFIX_RE.search(name)
        suffix = match.group(1).strip() if match else ""
        # Model ids such as "Alibaba qwen3.8-max" are not languages.
        if (
            suffix
            and is_foreign_language(suffix)
            and len(suffix) < 40
            and not any(ch.isdigit() for ch in suffix)
        ):
            foreign.append(suffix)
            continue
        claim_lang = str(
            finding.get("prompt_language") or finding.get("language") or ""
        ).strip()
        if len(models) == 1 and is_foreign_language(claim_lang):
            foreign.append(claim_lang)
            continue
        return ""
    if not foreign:
        return ""
    first = foreign[0]
    if any(lang.lower() != first.lower() for lang in foreign[1:]):
        return ", ".join(dict.fromkeys(foreign))
    return first


def evidence_status(finding: dict[str, Any]) -> str:
    """Reader-facing evidence ladder. One label per finding.

    Externally verified → Cross-model corroborated → Single-model lead → Contested.
    A source URL is the top rung. Disagreement is its own rung, not a weaker score.
    """
    status = str(finding.get("status") or "").upper().replace("_", "-").replace(" ", "-")
    if status == "CONTESTED":
        return "Contested"
    if has_source_url(finding):
        return "Externally verified"
    try:
        n_models = int(finding.get("corroboration") or 1)
    except (TypeError, ValueError):
        n_models = 1
    if n_models >= 2:
        return "Cross-model corroborated"
    return "Single-model lead"


def research_significance(finding: dict[str, Any]) -> str:
    """Reader-facing stakes: High, Medium, or Low.

    Collapses the internal sensitivity score. It is not an evidence status.
    """
    try:
        sens = int(finding.get("sensitivity") or 0)
    except (TypeError, ValueError):
        sens = 0
    if sens >= 4:
        return "High"
    if sens >= 3:
        return "Medium"
    return "Low"


def significance_band(finding: dict[str, Any]) -> str:
    return research_significance(finding).lower()


def provenance_label(finding: dict[str, Any], audience: str | None = None) -> str:
    """Same-line evidence status. Language stays on its own badge."""
    del audience
    return evidence_status(finding)


def page_eligible(finding: dict[str, Any], audience: str | None = None) -> bool:
    """One-pager and lead bar: a real URL, and not a single-model damaging claim."""
    if not has_source_url(finding):
        return False
    voice = normalize_audience(audience)
    if voice != OPPOSITION:
        return True
    try:
        n_models = int(finding.get("corroboration") or 1)
    except (TypeError, ValueError):
        n_models = 1
    if disclosure_class(finding, OPPOSITION) == "Damaging" and n_models <= 1:
        return False
    return True


def claim_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {word for word in words if len(word) > 2 and word not in _STOP}


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text or ""))


def group_episodes(findings: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group facets of one allegation, and near-duplicate wording, into episodes."""
    if not findings:
        return []
    keys: list[str] = []
    by_key: dict[str, dict[str, Any]] = {}
    for index, finding in enumerate(findings):
        cid = str(finding.get("claim_id") or "").strip() or f"row{index}"
        key = cid if cid not in by_key else f"{cid}#{index}"
        keys.append(key)
        by_key[key] = finding

    parent = {key: key for key in keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    misconduct = [
        key
        for key in keys
        if _MISCONDUCT_RE.search(str(by_key[key].get("claim") or ""))
    ]
    for earlier, later in zip(misconduct, misconduct[1:]):
        union(earlier, later)

    token_sets = {key: claim_tokens(str(by_key[key].get("claim") or "")) for key in keys}
    for index, left in enumerate(keys):
        left_tokens = token_sets[left]
        if len(left_tokens) < 4:
            continue
        for right in keys[index + 1 :]:
            if find(left) == find(right):
                continue
            right_tokens = token_sets[right]
            if len(right_tokens) < 4:
                continue
            if _numbers(str(by_key[left].get("claim") or "")) != _numbers(
                str(by_key[right].get("claim") or "")
            ):
                continue
            overlap = len(left_tokens & right_tokens)
            if not overlap:
                continue
            if overlap / len(left_tokens | right_tokens) >= 0.55:
                union(left, right)

    order: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for key in keys:
        root = find(key)
        if root not in groups:
            groups[root] = []
            order.append(root)
        groups[root].append(by_key[key])
    return [groups[root] for root in order]


def presentation_rows(
    findings: list[dict[str, Any]],
    audience: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split episodes into sourced lead rows and single-model / unverified rows.

    Each returned row is one episode. ``facets`` holds the other claims in it.
    """
    shown: list[dict[str, Any]] = []
    parked: list[dict[str, Any]] = []
    for members in group_episodes(findings):
        eligible = [member for member in members if page_eligible(member, audience)]
        pool = eligible or members
        rep = dict(pool[0])
        rep_id = rep.get("claim_id")
        rep["facets"] = [
            member for member in members if member.get("claim_id") != rep_id
        ]
        if eligible:
            shown.append(rep)
        else:
            parked.append(rep)
    return shown, parked


def slim_finding(finding: dict[str, Any]) -> dict[str, Any]:
    """Compact finding for the executive-summary prompt."""
    citations = [
        str(item)
        for item in (finding.get("citations") or [])
        if _URL_RE.search(str(item or ""))
    ][:4]
    return {
        "claim_id": finding.get("claim_id"),
        "claim": finding.get("claim"),
        "status": finding.get("status"),
        "provenance": finding.get("provenance") or provenance_label(
            finding, finding.get("audience")
        ),
        "sensitivity": finding.get("sensitivity"),
        "corroboration": finding.get("corroboration"),
        "source_model": finding.get("source_model"),
        "citations": citations,
        "facets": [member.get("claim") for member in (finding.get("facets") or [])][:8],
    }
