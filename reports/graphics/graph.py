"""Evidence graph SVG: models → claims → chains."""

from __future__ import annotations

from typing import Iterable

from .style import (
    CREAM,
    CREAM_DEEP,
    FONT,
    FONT_MONO,
    INK,
    MUTED,
    PRINT_MAX_HEIGHT,
    PRINT_MAX_WIDTH,
    RULE,
    TEAL,
    TEAL_DEEP,
    TEAL_SOFT,
    WHITE,
    escape_xml,
    short_model_name,
    svg_root,
    title_block,
    truncate,
)


def _curve(x1: float, y1: float, x2: float, y2: float, *, opacity: float, color: str) -> str:
    # Stronger horizontal bend so curves read clearly in print
    dx = max(36.0, abs(x2 - x1) * 0.45)
    c1x = x1 + dx
    c2x = x2 - dx
    return (
        f'<path d="M {x1:.1f},{y1:.1f} C {c1x:.1f},{y1:.1f} {c2x:.1f},{y2:.1f} {x2:.1f},{y2:.1f}" '
        f'fill="none" stroke="{color}" stroke-opacity="{opacity:.2f}" stroke-width="1.6"/>'
    )


def _pick_findings(
    findings: list[dict],
    chains: list[dict],
    max_nodes: int,
) -> list[dict]:
    """Prefer claims that appear in chains so model→claim→chain edges exist."""
    by_id = {f["claim_id"]: f for f in findings if f.get("claim_id")}
    picked: list[dict] = []
    seen: set[str] = set()
    for ch in chains:
        for cid in ch.get("claim_ids") or []:
            if cid in by_id and cid not in seen:
                picked.append(by_id[cid])
                seen.add(cid)
            if len(picked) >= max_nodes:
                return picked
    for f in findings:
        cid = f.get("claim_id")
        if cid and cid not in seen:
            picked.append(f)
            seen.add(cid)
        if len(picked) >= max_nodes:
            break
    return picked


def evidence_graph_svg(
    findings: Iterable[dict],
    chains: Iterable[dict],
    *,
    max_nodes: int = 10,
    aliases: dict[str, str] | None = None,
    max_width: float = PRINT_MAX_WIDTH,
    max_height: float = PRINT_MAX_HEIGHT,
) -> str:
    """Sankey-style graph sized to keep node labels inside the cream columns."""
    all_findings = list(findings)
    chains = list(chains)[:4]
    width = min(max_width, 640)
    height = min(max_height, 400)

    # Title outside columns; column titles sit inside the beige bands at the bottom.
    band_top = 56
    band_bottom = height - 14
    band_h = band_bottom - band_top
    footer_h = 22  # MODELS / CLAIMS / CHAINS row inside each band
    # Vertical room for labels above models and below chain names.
    node_top = band_top + 28
    node_bot = band_bottom - footer_h - 30

    fit_nodes = min(max_nodes, max(6, int((node_bot - node_top) / 28)))
    findings = _pick_findings(all_findings, chains, fit_nodes)

    x_model, x_claim, x_chain = 96, 320, 560
    col_w = 118

    models: list[str] = []
    seen_m: set[str] = set()
    for f in findings:
        m = short_model_name(f["source_model"], aliases)
        if m not in seen_m:
            seen_m.add(m)
            models.append(m)

    def _ys(n: int, top: float = node_top, bot: float = node_bot) -> list[float]:
        if n <= 1:
            return [(top + bot) / 2]
        return [top + i * (bot - top) / (n - 1) for i in range(n)]

    my = _ys(len(models) or 1)
    cy = _ys(len(findings) or 1)
    chy = _ys(len(chains) or 1)

    model_pos = {m: (x_model, my[i]) for i, m in enumerate(models)}
    claim_pos = {f["claim_id"]: (x_claim, cy[i]) for i, f in enumerate(findings)}
    chain_pos = {c["chain_id"]: (x_chain, chy[i]) for i, c in enumerate(chains)}
    chain_label = {
        c["chain_id"]: truncate(c.get("label") or c["chain_id"], 22) for c in chains
    }

    bands = []
    for x, label in (
        (x_model, "Models"),
        (x_claim, "Claims"),
        (x_chain, "Chains"),
    ):
        bands.append(
            f'<rect x="{x - col_w / 2:.1f}" y="{band_top}" width="{col_w}" '
            f'height="{band_h}" rx="12" fill="{CREAM}" stroke="{RULE}"/>'
            f'<text x="{x:.1f}" y="{band_bottom - 8:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="10" font-weight="600" letter-spacing="0.06em" '
            f'fill="{MUTED}">{label.upper()}</text>'
        )

    edges = []
    for f in findings:
        m = short_model_name(f["source_model"], aliases)
        if m in model_pos and f["claim_id"] in claim_pos:
            x1, y1 = model_pos[m]
            x2, y2 = claim_pos[f["claim_id"]]
            edges.append(_curve(x1 + 14, y1, x2 - 36, y2, opacity=0.5, color=TEAL))

    for ch in chains:
        for cid in ch.get("claim_ids", [])[:10]:
            if cid in claim_pos and ch["chain_id"] in chain_pos:
                x1, y1 = claim_pos[cid]
                x2, y2 = chain_pos[ch["chain_id"]]
                edges.append(_curve(x1 + 36, y1, x2 - 14, y2, opacity=0.4, color=INK))

    nodes = []
    for m, (x, y) in model_pos.items():
        nodes.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{TEAL}" stroke="{TEAL_DEEP}" '
            f'stroke-width="1.5"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{WHITE}"/>'
            f'<text x="{x:.1f}" y="{y - 16:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="10" font-weight="600" fill="{INK}">'
            f"{escape_xml(truncate(m, 12))}</text>"
        )

    for cid, (x, y) in claim_pos.items():
        nodes.append(
            f'<g>'
            f'<title>{escape_xml(cid)}</title>'
            f'<rect x="{x - 34:.1f}" y="{y - 11:.1f}" width="68" height="22" rx="5" '
            f'fill="{WHITE}" stroke="{TEAL_DEEP}" stroke-width="1.25"/>'
            f'<text x="{x:.1f}" y="{y + 1:.1f}" text-anchor="middle" '
            f'font-family="{FONT_MONO}" font-size="9" font-weight="600" fill="{INK}">'
            f"{escape_xml(cid)}</text>"
            f"</g>"
        )

    for chid, (x, y) in chain_pos.items():
        label = chain_label.get(chid, chid)
        nodes.append(
            f'<g>'
            f'<title>{escape_xml(label)}</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="{INK}" stroke="{CREAM_DEEP}" '
            f'stroke-width="1.75"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{TEAL_SOFT}"/>'
            f'<text x="{x:.1f}" y="{y - 16:.1f}" text-anchor="middle" '
            f'font-family="{FONT_MONO}" font-size="9" font-weight="600" fill="{INK}">'
            f"{escape_xml(chid)}</text>"
            f'<text x="{x:.1f}" y="{y + 24:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="7.5" fill="{MUTED}">'
            f"{escape_xml(label)}</text>"
            f"</g>"
        )

    body = f"""  {title_block(width / 2, 20, "Evidence Graph")}
  {"".join(bands)}
  {"".join(edges)}
  {"".join(nodes)}"""
    return svg_root(width, height, body)
