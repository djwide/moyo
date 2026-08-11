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
BLACK = "#000000"

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

# Severity palette for distribution bars
BAR_COLORS = {
    "high": INK,
    "medium": TEAL_DEEP,
    "low": TEAL,
    "informational": "#C4BFB2",
}

BAR_LABELS = {
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "informational": "Info",
}


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


def full_model_name(source_model: str) -> str:
    """Full model label for charts: keep vendor/model detail, drop language suffix.

    ``ChatGPT (OpenAI gpt-4o) (French)`` → ``ChatGPT (OpenAI gpt-4o)``
    ``Llama 4 Maverick (French)`` → ``Llama 4 Maverick``
    """
    raw = " ".join(str(source_model or "").split()).strip()
    if not raw:
        return "unknown"
    if raw.endswith(")") and " (" in raw:
        base, last = raw.rsplit(" (", 1)
        lang = last[:-1].strip()
        # Language tags are alphabetic / spaces (e.g. Mandarin Chinese), not model ids.
        if lang and not any(ch.isdigit() for ch in lang) and len(lang) < 40:
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
    """Ensure root SVG uses viewBox + width=100% for graphic-box fitting."""

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
            f'width="100%" preserveAspectRatio="xMidYMid meet" role="img">'
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
