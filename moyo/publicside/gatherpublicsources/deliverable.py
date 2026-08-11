"""Build a displayable Deliverable HTML report via the Grok (xAI) API.

The Deliverable is a four-section synthesis over an exploration report and its
claims brief (``summary.md``):

1. Executive Exposure Summary
2. Evidence Graph
3. Findings & Basis Chains
4. Mitigation Playbook

Primary artifact is ``deliverable.html`` (open in a browser). A companion
``deliverable.md`` sidecar is written only when the model returns markdown.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

from moyo.llm.client import LLMClient, LLMSpec, ensure_env_loaded

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]

DEFAULT_DELIVERABLE_MODEL = "grok-4.5"
DEFAULT_DELIVERABLE_BASE_URL = "https://api.x.ai/v1"
DEFAULT_DELIVERABLE_MAX_TOKENS = 8000
DEFAULT_DELIVERABLE_TIMEOUT = 300
# Cap how much exploration body we ship to Grok alongside summary.md.
DEFAULT_EXPLORATION_CHARS = 48000

DELIVERABLE_SYSTEM = (
    "You are a senior exposure analyst writing a formal Deliverable for "
    "decision-makers. You turn multi-LLM retrieval findings and a claims brief "
    "into a precise, actionable, browser-displayable HTML report. Prefer "
    "concrete specifics (numbers, dates, named entities, source labels) over "
    "vague generalities. Never invent sources, citations, or facts that are "
    "not supported by the supplied material. When sources disagree, state both "
    "sides with attribution. Output a single complete HTML document only — "
    "no markdown fences, no preamble."
)


@dataclass
class DeliverableResult:
    """Outcome of Deliverable synthesis."""

    prompt: str
    markdown: str
    html: str = ""
    output_path: Optional[str] = None
    html_path: Optional[str] = None
    summary_path: Optional[str] = None
    exploration_path: Optional[str] = None


def get_deliverable_llm(override: Optional[LLMClient] = None) -> LLMClient:
    """Return the Grok (xAI) client used to author Deliverables.

    Override with a configured client, or via ``MOYO_DELIVERABLE_MODEL`` /
    ``MOYO_DELIVERABLE_BASE_URL`` / ``MOYO_DELIVERABLE_API_KEY`` (falls back to
    ``XAI_API_KEY``).
    """
    if override is not None:
        return override

    ensure_env_loaded()
    model = (
        os.environ.get("MOYO_DELIVERABLE_MODEL") or DEFAULT_DELIVERABLE_MODEL
    ).strip()
    base_url = (
        os.environ.get("MOYO_DELIVERABLE_BASE_URL") or DEFAULT_DELIVERABLE_BASE_URL
    ).strip()
    api_key = (
        os.environ.get("MOYO_DELIVERABLE_API_KEY")
        or os.environ.get("XAI_API_KEY")
        or ""
    ).strip() or None
    try:
        max_tokens = int(
            os.environ.get("MOYO_DELIVERABLE_MAX_TOKENS")
            or DEFAULT_DELIVERABLE_MAX_TOKENS
        )
    except ValueError:
        max_tokens = DEFAULT_DELIVERABLE_MAX_TOKENS
    try:
        timeout = int(
            os.environ.get("MOYO_DELIVERABLE_TIMEOUT") or DEFAULT_DELIVERABLE_TIMEOUT
        )
    except ValueError:
        timeout = DEFAULT_DELIVERABLE_TIMEOUT

    if not api_key:
        raise RuntimeError(
            "Deliverable synthesis requires an xAI API key "
            "(set XAI_API_KEY or MOYO_DELIVERABLE_API_KEY)"
        )

    return LLMClient(
        LLMSpec(
            provider="custom",
            model=model,
            base_url=base_url,
            api_key=api_key,
            label=f"Grok (xAI {model}) (deliverable)",
            temperature=0.3,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    )


def _prompt_from_exploration(exploration_md: str) -> str:
    match = re.search(
        r"^#\s+Topic exploration:\s*(.+?)\s*$", exploration_md, flags=re.M
    )
    if match:
        return match.group(1).strip()
    match = re.search(r"^#\s+Claims summary:\s*(.+?)\s*$", exploration_md, flags=re.M)
    if match:
        return match.group(1).strip()
    return "Untitled topic"


def _truncate(text: str, limit: int) -> str:
    body = (text or "").strip()
    if len(body) <= limit:
        return body
    return body[:limit].rstrip() + "\n\n... [truncated for Deliverable synthesis]"


def _deliverable_ask(prompt: str, summary_md: str, exploration_md: str) -> str:
    try:
        explor_limit = int(
            os.environ.get("MOYO_DELIVERABLE_EXPLORATION_CHARS")
            or DEFAULT_EXPLORATION_CHARS
        )
    except ValueError:
        explor_limit = DEFAULT_EXPLORATION_CHARS

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f'Topic / original ask: "{prompt}"\n\n'
        "Using ONLY the claims brief and exploration excerpts below, write a "
        "formal Deliverable as ONE complete, self-contained HTML5 document "
        "that opens cleanly in a browser. Requirements:\n\n"
        "1. Start with `<!DOCTYPE html>` and end with `</html>`. No markdown "
        "fences, no text before/after the HTML document.\n"
        "2. Include `<meta charset=\"utf-8\">`, a viewport meta, and an "
        f"`<title>Deliverable: {prompt}</title>`.\n"
        "3. Embed CSS in a `<style>` block (no external stylesheets required). "
        "Use a clear, readable light or dark theme with distinct section "
        "headers, tables, and source chips/badges. Avoid purple-on-white "
        "generic AI aesthetics.\n"
        "4. Body must contain exactly these sections with these headings:\n"
        "   - `<h1>Deliverable: {prompt}</h1>`\n"
        f"   - a small meta line: Generated {stamp} via Grok (xAI)\n"
        "   - `<h2>§ 1 Executive Exposure Summary</h2>` — exposure count by "
        "sensitivity, trend, and the three chains that matter most\n"
        "   - `<h2>§ 2 Evidence Graph</h2>` — signals and inferences; every "
        "node a source, every edge a reproducible reasoning step (node list + "
        "edge list; optional mermaid if compact)\n"
        "   - `<h2>§ 3 Findings & Basis Chains</h2>` — inference, confidence, "
        "sensitivity, full source attribution\n"
        "   - `<h2>§ 4 Mitigation Playbook</h2>` — prioritized reword / retime "
        "/ redact / restructure actions, plus a Method appendix\n"
        "5. Use exact Source labels from the brief. Do not invent facts.\n"
        "6. Prefer points of precision and explicit disagreements.\n"
        "7. Keep it focused for a busy reader; finish the full HTML document "
        "(do not stop mid-tag).\n\n"
        "=== CLAIMS BRIEF (summary.md) ===\n"
        f"{summary_md.strip()}\n\n"
        "=== EXPLORATION EXCERPT (exploration.md) ===\n"
        f"{_truncate(exploration_md, explor_limit)}\n"
    )


def _extract_html(text: str) -> Optional[str]:
    body = (text or "").strip()
    if not body:
        return None
    # Strip fences if the model wrapped the document.
    if body.startswith("```"):
        body = re.sub(r"^```(?:html|HTML)?\s*", "", body)
        body = re.sub(r"\s*```$", "", body).strip()
    match = re.search(r"<!DOCTYPE html\b.*</html\s*>", body, flags=re.S | re.I)
    if match:
        return match.group(0).strip()
    match = re.search(r"<html\b.*</html\s*>", body, flags=re.S | re.I)
    if match:
        return "<!DOCTYPE html>\n" + match.group(0).strip()
    # Truncated HTML: keep from DOCTYPE/html and close conservatively.
    start = re.search(r"<!DOCTYPE html\b|<html\b", body, flags=re.I)
    if not start:
        return None
    html = body[start.start() :].strip()
    html = re.sub(r"<[^>]*$", "", html)
    if "</body>" not in html.lower():
        html += "\n</body>"
    if "</html>" not in html.lower():
        html += "\n</html>"
    if not html.lstrip().lower().startswith("<!doctype"):
        html = "<!DOCTYPE html>\n" + html
    return html


def _markdown_from_html_fallback(prompt: str, html: str, label: str) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"# Deliverable: {prompt}\n\n"
        f"_Generated {stamp} via {label}_\n\n"
        "_Open `deliverable.html` in a browser for the displayable report._\n"
    )


def synthesize_deliverable(
    prompt: str,
    summary_md: str,
    exploration_md: str,
    llm: Optional[LLMClient] = None,
) -> Tuple[str, str]:
    """Call Grok to author the Deliverable.

    Returns ``(html, markdown_sidecar)``.
    """
    llm = get_deliverable_llm(llm)
    ask = _deliverable_ask(prompt, summary_md, exploration_md)
    text = llm.complete(
        ask,
        system=DELIVERABLE_SYSTEM,
        max_tokens=llm.spec.max_tokens,
    )
    if not text or not str(text).strip():
        raise RuntimeError("Deliverable synthesis returned no content")
    body = str(text).strip()
    html = _extract_html(body)
    if not html:
        # Model returned markdown — wrap in a minimal displayable HTML shell.
        md = body
        if md.startswith("```"):
            md = re.sub(r"^```(?:markdown)?\s*", "", md)
            md = re.sub(r"\s*```$", "", md).strip()
        if not md.lstrip().startswith("#"):
            md = f"# Deliverable: {prompt}\n\n{md}"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta = f"_Generated {stamp} via {llm.label}_\n\n"
        if md.startswith("# "):
            first_nl = md.find("\n")
            md = md[: first_nl + 1] + "\n" + meta + md[first_nl + 1 :].lstrip("\n")
        else:
            md = meta + md
        escaped = (
            md.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        html = (
            "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>Deliverable: {prompt}</title>"
            "<style>body{font-family:Georgia,serif;max-width:52rem;margin:2rem auto;"
            "padding:0 1.25rem;line-height:1.55;color:#1a1a1a;background:#faf9f7}"
            "pre{white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:0.92rem}"
            "</style></head><body>"
            f"<pre>{escaped}</pre></body></html>\n"
        )
        return html, md

    sidecar = _markdown_from_html_fallback(prompt, html, llm.label)
    return html, sidecar


def build_deliverable(
    exploration_path: str | Path,
    summary_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    llm: Optional[LLMClient] = None,
    progress: Optional[ProgressFn] = None,
) -> DeliverableResult:
    """Build ``deliverable.html`` (and a short ``deliverable.md`` pointer)."""

    def _report(msg: str) -> None:
        logger.info(msg)
        if progress:
            progress(msg)

    explor_path = Path(exploration_path)
    if not explor_path.is_file():
        raise FileNotFoundError(f"exploration report not found: {explor_path}")

    sum_path = Path(summary_path) if summary_path else explor_path.parent / "summary.md"
    if not sum_path.is_file():
        raise FileNotFoundError(
            f"claims summary not found: {sum_path} "
            "(run `moyo-gather summarize` first)"
        )

    exploration_md = explor_path.read_text(encoding="utf-8")
    summary_md = sum_path.read_text(encoding="utf-8")
    prompt = _prompt_from_exploration(exploration_md)
    if prompt == "Untitled topic":
        prompt = _prompt_from_exploration(summary_md)

    client = get_deliverable_llm(llm)
    _report(
        f"Building Deliverable via {client.label} from {sum_path.name} "
        f"+ {explor_path.name} ..."
    )
    html, markdown = synthesize_deliverable(
        prompt, summary_md, exploration_md, llm=client
    )

    if output_path:
        out = Path(output_path)
        html_out = out if out.suffix.lower() in {".html", ".htm"} else out.with_suffix(".html")
        md_out = html_out.with_suffix(".md")
    else:
        html_out = explor_path.parent / "deliverable.html"
        md_out = explor_path.parent / "deliverable.md"

    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(html.rstrip() + "\n", encoding="utf-8")
    md_out.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    _report(f"Wrote Deliverable HTML to {html_out}")
    _report(f"Wrote Deliverable sidecar to {md_out}")

    return DeliverableResult(
        prompt=prompt,
        markdown=markdown,
        html=html,
        output_path=str(md_out),
        html_path=str(html_out),
        summary_path=str(sum_path),
        exploration_path=str(explor_path),
    )
