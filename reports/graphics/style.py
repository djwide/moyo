"""Shared MOYO brand tokens for report SVG figures."""

from __future__ import annotations

import re

TEAL = "#4FB0A2"
TEAL_DEEP = "#2F7A70"
TEAL_SOFT = "#A8D4CB"
CREAM = "#F2F1E8"
CREAM_DEEP = "#E6E2D6"
INK = "#1D2228"
MUTED = "#5C6570"
RULE = "#D9D4C8"
WHITE = "#FFFFFF"
BLACK = "#2E353D"

FONT = "IBM Plex Sans, 'Helvetica Neue', Helvetica, Arial, sans-serif"
FONT_MONO = "IBM Plex Mono, ui-monospace, Menlo, Consolas, monospace"

# Approximate glyph metrics for chart label type (~11–12px).
LETTER_H = 12
LETTER_W = 8

# A4 content frame targets (px @ ~96dpi). Keep charts inside these so PDF
# graphic boxes don't clip labels or overflow the page.
PRINT_MAX_WIDTH = 640
PRINT_MAX_HEIGHT = 400
PRINT_ONEPAGE_MAX_HEIGHT = 280

# Discrete sensitivity 0–5 (0 = empty cell)
HEAT_SCALE = {
    0: CREAM,
    1: "#D8EBE7",
    2: TEAL_SOFT,
    3: TEAL,
    4: TEAL_DEEP,
    5: INK,
}

# Disclosure palette for distribution bars. Legacy sensitivity keys still color
# older report JSON.
BAR_COLORS = {
    "security_relevant": INK,
    "damaging": INK,
    "potentially_damaging": TEAL_DEEP,
    "sensitive": INK,
    "commercially_sensitive": INK,
    "potentially_strategic": TEAL_DEEP,
    "material": TEAL_DEEP,
    "unexpected": TEAL_DEEP,
    "interesting": TEAL,
    "expected": "#C4BFB2",
    "high": INK,
    "medium": TEAL_DEEP,
    "low": TEAL,
    "informational": "#C4BFB2",
}

BAR_LABELS = {
    "security_relevant": "Security relevant",
    "damaging": "Damaging",
    "potentially_damaging": "Potentially damaging",
    "sensitive": "Sensitive",
    "commercially_sensitive": "Commercially sensitive",
    "potentially_strategic": "Potentially strategic",
    "material": "Material",
    "unexpected": "Unexpected",
    "interesting": "Interesting",
    "expected": "Expected",
    "high": "Security relevant",
    "medium": "Interesting",
    "low": "Interesting",
    "informational": "Expected",
}

DISCLOSURE_CHART_ORDER = ("security_relevant", "unexpected", "interesting", "expected")


def escape_xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def truncate(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= n:
        return text
    return text[: max(0, n - 1)].rstrip() + "…"


def raw_finding_models(finding: dict) -> list[str]:
    """Full source labels on a finding (``source_models``, else ``source_model``)."""
    raw_models = finding.get("source_models") if isinstance(finding, dict) else None
    if not isinstance(raw_models, list) or not raw_models:
        raw_models = [(finding or {}).get("source_model") or ""]
    return [str(m).strip() for m in raw_models if str(m).strip()]


def models_with_results(
    findings,
    *,
    models_probed=None,
    aliases: dict[str, str] | None = None,
) -> list[str]:
    """Full labels of models to chart, in roster order.

    When ``models_probed`` is set it is the whitelist (the scan roster).
    Silent probed models are kept so charts reflect the full response corpus
    rather than only models that produced extracted claims. Labels that only
    appear on findings and were never probed are omitted when the whitelist
    is set.
    """
    aliases = aliases or {}
    present: set[str] = set()
    for finding in findings or []:
        for raw in raw_finding_models(finding):
            key = short_model_name(raw, aliases)
            if key and key != "unknown":
                present.add(key)

    probed = [str(m).strip() for m in (models_probed or []) if str(m).strip()]
    source = probed if probed else [
        raw for finding in (findings or []) for raw in raw_finding_models(finding)
    ]
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in source:
        key = short_model_name(raw, aliases)
        if not key or key == "unknown" or key in seen:
            continue
        if probed:
            seen.add(key)
            ordered.append(raw)
            continue
        if key not in present:
            continue
        seen.add(key)
        ordered.append(raw)
    return ordered


def short_model_name(source_model: str, aliases: dict[str, str] | None = None) -> str:
    """Map a finding's source_model to a short display label.

    Handles exact alias hits, language-suffixed labels
    (``ChatGPT (OpenAI gpt-4o) (French)``), and a plain head-name fallback.
    """
    aliases = aliases or {}
    raw = (source_model or "").strip()
    if not raw:
        return "unknown"
    if raw in aliases:
        return aliases[raw]
    # Strip trailing " (Language)" once so aliases keyed on the base label hit.
    if raw.endswith(")") and " (" in raw:
        base = raw.rsplit(" (", 1)[0].strip()
        if base in aliases:
            return aliases[base]
        # Prefer the human head before the first parenthetical vendor tag
        head = base.split("(", 1)[0].strip() or base
        return head
    return raw.split("(", 1)[0].strip() or raw


_LANGUAGE_SUFFIXES = frozenset(
    {
        "arabic",
        "bengali",
        "chinese",
        "czech",
        "danish",
        "dutch",
        "english",
        "farsi",
        "finnish",
        "french",
        "german",
        "greek",
        "hebrew",
        "hindi",
        "hungarian",
        "indonesian",
        "italian",
        "japanese",
        "korean",
        "malay",
        "mandarin",
        "mandarin chinese",
        "norwegian",
        "persian",
        "polish",
        "portuguese",
        "romanian",
        "russian",
        "simplified chinese",
        "spanish",
        "swedish",
        "tagalog",
        "tamil",
        "thai",
        "traditional chinese",
        "turkish",
        "ukrainian",
        "urdu",
        "vietnamese",
    }
)


def full_model_name(source_model: str) -> str:
    """Full model label: keep vendor/model id, drop a trailing language suffix.

    ``ChatGPT (OpenAI gpt-4o) (French)`` → ``ChatGPT (OpenAI gpt-4o)``
    ``Claude (Anthropic Sonnet)`` → ``Claude (Anthropic Sonnet)``
    ``Llama 4 Maverick (French)`` → ``Llama 4 Maverick``
    """
    raw = " ".join(str(source_model or "").split()).strip()
    if not raw:
        return "unknown"
    if raw.endswith(")") and " (" in raw:
        base, last = raw.rsplit(" (", 1)
        lang = last[:-1].strip()
        if lang.lower() in _LANGUAGE_SUFFIXES:
            return base.strip() or raw
    return raw


def format_source_cite(
    source_model: str,
    *,
    corroboration: int | None = None,
    peer_models: list[str] | None = None,
    aliases: dict[str, str] | None = None,
) -> str:
    """Primary model name plus ``+ N`` for other corroborating models.

    Examples: ``Kimi``, ``Kimi + 5``.
    """
    aliases = aliases or {}
    primary = short_model_name(source_model, aliases)
    if peer_models is not None:
        peers = {short_model_name(m, aliases) for m in peer_models if m}
        others = max(0, len(peers) - (1 if primary in peers else 0))
    else:
        others = max(0, int(corroboration or 1) - 1)
    if others:
        return f"{primary} + {others}"
    return primary


def svg_root(width: float, height: float, body: str, *, bg: str = WHITE) -> str:
    """Build an SVG that scales into print/HTML boxes via viewBox.

    Fixed pixel width/height are omitted so CSS ``max-width`` / ``max-height``
    can shrink the figure without clipping. ``preserveAspectRatio`` keeps
    labels readable when scaled.
    """
    w, h = max(1.0, float(width)), max(1.0, float(height))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w:.0f} {h:.0f}" width="100%" '
        f'preserveAspectRatio="xMidYMid meet" role="img">\n'
        f'  <rect width="100%" height="100%" fill="{bg}"/>\n'
        f"{body}\n"
        f"</svg>\n"
    )


def fit_canvas(
    content_w: float,
    content_h: float,
    *,
    max_w: float = PRINT_MAX_WIDTH,
    max_h: float = PRINT_MAX_HEIGHT,
) -> tuple[float, float, float]:
    """Return ``(width, height, scale)`` capped to the print frame."""
    content_w = max(1.0, float(content_w))
    content_h = max(1.0, float(content_h))
    scale = min(1.0, max_w / content_w, max_h / content_h)
    return content_w * scale, content_h * scale, scale


def normalize_svg_for_embed(svg: str) -> str:
    """Ensure root SVG uses viewBox only; CSS owns width/height for print fit.

    WeasyPrint often treats ``width="100%"`` as the viewBox's pixel width, which
    overflows A4 when charts sit side-by-side. Omit width/height on the root so
    ``max-width: 100%`` in CSS can constrain them.
    """

    def _rewrite(match: re.Match[str]) -> str:
        tag = match.group(0)
        vb = re.search(r'viewBox="([^"]+)"', tag)
        wm = re.search(r'\bwidth="([\d.]+)"', tag)
        hm = re.search(r'\bheight="([\d.]+)"', tag)
        if vb:
            view = vb.group(1)
        elif wm and hm:
            view = f"0 0 {float(wm.group(1)):.0f} {float(hm.group(1)):.0f}"
        else:
            view = f"0 0 {PRINT_MAX_WIDTH} {PRINT_MAX_HEIGHT}"
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}" '
            f'preserveAspectRatio="xMidYMid meet" role="img" '
            f'style="width:100%;height:auto;max-width:100%;display:block">'
        )

    return re.sub(r"<svg\b[^>]*>", _rewrite, svg, count=1)


def title_block(x: float, y: float, title: str, *, subtitle: str | None = None) -> str:
    parts = [
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
        f'font-family="{FONT}" font-size="14" font-weight="600" fill="{INK}">'
        f"{escape_xml(title)}</text>"
    ]
    if subtitle:
        parts.append(
            f'<text x="{x:.1f}" y="{y + 16:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="10" fill="{MUTED}">'
            f"{escape_xml(subtitle)}</text>"
        )
    return "\n  ".join(parts)
