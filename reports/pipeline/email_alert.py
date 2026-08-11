"""Short sales/alert email highlighting highest-impact claims."""

from __future__ import annotations

from html import escape as html_escape
from pathlib import Path
from typing import Any

from .textclean import plain_text


def _cited_labels(finding: dict[str, Any], limit: int = 2) -> str:
    """Real-world sources behind one finding, as a short display string."""
    labels = [
        str(c.get("short") or c.get("label") or "").strip()
        for c in (finding.get("citations_display") or [])
        if str(c.get("short") or c.get("label") or "").strip()
    ]
    return "; ".join(labels[:limit])


def _impact_key(finding: dict[str, Any]) -> tuple:
    return (
        int(finding.get("sensitivity") or 0),
        int(finding.get("specificity") or 0),
        int(finding.get("interestingness") or 0),
        int(finding.get("novelty") or 0),
    )


def pick_impact_findings(
    findings: list[dict[str, Any]],
    *,
    limit: int = 3,
    min_sensitivity: int = 3,
) -> list[dict[str, Any]]:
    """Highest-impact findings for the alert email body."""
    ranked = sorted(findings, key=_impact_key, reverse=True)
    picks = [f for f in ranked if int(f.get("sensitivity") or 0) >= min_sensitivity]
    if not picks:
        picks = ranked
    return picks[:limit]


def build_alert_email(
    content: dict[str, Any],
    *,
    cta_text: str = "Request the full MOYO exposure assessment",
    cta_url: str | None = None,
    include_remediation: bool | None = None,
) -> dict[str, str]:
    """Return ``subject``, ``text``, and ``html`` for a short alert email."""
    meta = content.get("meta") or {}
    topic = meta.get("topic") or "your topic"
    headline = meta.get("headline") or topic
    date = meta.get("report_date") or ""
    counts = meta.get("counts") or {}
    top = content.get("top_finding") or {}
    findings = list(content.get("findings") or [])
    if include_remediation is None:
        include_remediation = bool(meta.get("include_remediation"))

    impact = pick_impact_findings(findings)
    # Always surface top_finding first if not already included
    top_id = top.get("claim_id")
    if top_id and top.get("text"):
        if not any(f.get("claim_id") == top_id for f in impact):
            impact = [
                {
                    "claim_id": top_id,
                    "claim": top.get("text"),
                    "sensitivity": None,
                    "specificity": None,
                    "status": (top.get("badges") or ["TOP"])[0],
                    "source_cite": "",
                },
                *impact,
            ][:3]

    n_findings = counts.get("findings", len(findings))
    n_high = counts.get("high_sensitivity", 0)
    n_models = counts.get("llms_tested", 0)

    subject = f"MOYO alert: high-impact exposure on {topic}"

    bullets_txt = []
    for f in impact:
        cite = f.get("source_cite") or f.get("source_short") or ""
        sens = f.get("sensitivity")
        tip = f" (sens {sens})" if sens is not None else ""
        src = f" — {cite}" if cite else ""
        claim = plain_text(f.get("claim") or f.get("text"))
        cited = _cited_labels(f)
        bullets_txt.append(
            f"- [{f.get('claim_id')}]{tip} {claim}{src}"
            + (f"\n  Cited: {cited}" if cited else "")
        )

    cta_line = cta_text
    if cta_url:
        cta_line = f"{cta_text}: {cta_url}"

    full_bits = [
        "complete finding index",
        "evidence excerpts with line references",
        "model heatmap",
        "cited real-world sources",
    ]
    if include_remediation:
        full_bits.append("a remediation playbook")
    full_list = ", ".join(full_bits[:-1]) + f", and {full_bits[-1]}"

    text = "\n".join(
        [
            f"MOYO exposure alert — {date}".strip(" —"),
            "",
            headline,
            "",
            f"We assessed {n_models} models and retained {n_findings} findings "
            f"({n_high} high-sensitivity).",
            "",
            "Highest-impact claims from this run:",
            *bullets_txt,
            "",
            f"This note is a teaser only. The full assessment includes the {full_list}.",
            "",
            cta_line,
            "",
            "— MOYO",
            "",
        ]
    )

    items_html = []
    for f in impact:
        cite = f.get("source_cite") or f.get("source_short") or ""
        sens = f.get("sensitivity")
        meta_bits = []
        if f.get("claim_id"):
            meta_bits.append(str(f["claim_id"]))
        if sens is not None:
            meta_bits.append(f"sens {sens}")
        if cite:
            meta_bits.append(str(cite))
        meta_line = html_escape(" · ".join(meta_bits))
        claim = html_escape(plain_text(f.get("claim") or f.get("text")))
        cited = _cited_labels(f)
        cited_html = (
            f"<div style=\"font-family:Helvetica,Arial,sans-serif;font-size:11px;"
            f"color:#2f7a70;margin-top:3px;\">Cited: {html_escape(cited)}</div>"
            if cited
            else ""
        )
        items_html.append(
            "<li style=\"margin:0 0 12px;\">"
            f"<div style=\"font-family:ui-monospace,Menlo,Consolas,monospace;"
            f"font-size:12px;color:#5c6570;\">{meta_line}</div>"
            f"<div style=\"font-family:Georgia,'Times New Roman',serif;"
            f"font-size:15px;color:#1d2228;margin-top:4px;\">{claim}</div>"
            f"{cited_html}"
            "</li>"
        )

    # Email may include a CTA URL; the PDF snapshot must not.
    if cta_url:
        cta_html = (
            f'<p style="margin:20px 0 0;">'
            f'<a href="{cta_url}" style="display:inline-block;background:#4fb0a2;'
            f'color:#000;text-decoration:none;font-weight:700;font-size:13px;'
            f'letter-spacing:0.06em;text-transform:uppercase;padding:10px 14px;">'
            f"{cta_text}</a></p>"
        )
    else:
        cta_html = (
            f'<p style="margin:20px 0 0;font-weight:700;color:#2f7a70;'
            f'font-size:13px;letter-spacing:0.06em;text-transform:uppercase;">'
            f"{cta_text}</p>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8" /><title>{subject}</title></head>
<body style="margin:0;padding:0;background:#f2f1e8;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#f2f1e8;">
    <tr><td align="center" style="padding:24px 12px;">
      <table role="presentation" width="560" cellspacing="0" cellpadding="0"
             style="background:#ffffff;border:1px solid #d9d4c8;max-width:560px;">
        <tr>
          <td style="background:#000;color:#f2f1e8;padding:14px 18px;
                     font-family:Helvetica,Arial,sans-serif;font-size:14px;
                     font-weight:700;letter-spacing:0.16em;">
            MOYO
            <span style="float:right;color:#4fb0a2;font-size:11px;
                         letter-spacing:0.1em;font-weight:600;">EXPOSURE ALERT</span>
          </td>
        </tr>
        <tr>
          <td style="padding:22px 18px 8px;font-family:Helvetica,Arial,sans-serif;">
            <div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
                        color:#2f7a70;margin-bottom:8px;">{date}</div>
            <h1 style="margin:0 0 12px;font-size:22px;line-height:1.2;color:#1d2228;
                       letter-spacing:-0.02em;">{headline}</h1>
            <p style="margin:0 0 16px;font-family:Georgia,'Times New Roman',serif;
                      font-size:15px;color:#1d2228;line-height:1.45;">
              We assessed <strong>{n_models}</strong> models and retained
              <strong>{n_findings}</strong> findings
              (<strong>{n_high}</strong> high-sensitivity). Below are the
              highest-impact claims from this run.
            </p>
            <ol style="margin:0;padding-left:18px;">
              {"".join(items_html)}
            </ol>
            <p style="margin:18px 0 0;font-family:Georgia,'Times New Roman',serif;
                      font-size:14px;color:#5c6570;line-height:1.45;">
              This note is a teaser. The full assessment includes the {full_list}.
            </p>
            {cta_html}
          </td>
        </tr>
        <tr>
          <td style="padding:14px 18px 18px;font-family:Helvetica,Arial,sans-serif;
                     font-size:11px;color:#5c6570;">
            — MOYO exposure assessment
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
"""
    return {"subject": subject, "text": text, "html": html}


def write_alert_email(
    content: dict[str, Any],
    out_dir: Path,
    *,
    cta_text: str = "Request the full MOYO exposure assessment",
    cta_url: str | None = None,
    include_remediation: bool | None = None,
) -> dict[str, Path]:
    """Write ``alert-email.*`` artifacts under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_alert_email(
        content,
        cta_text=cta_text,
        cta_url=cta_url or None,
        include_remediation=include_remediation,
    )
    paths = {
        "subject": out_dir / "alert-email.subject.txt",
        "text": out_dir / "alert-email.txt",
        "html": out_dir / "alert-email.html",
    }
    paths["subject"].write_text(payload["subject"] + "\n", encoding="utf-8")
    paths["text"].write_text(payload["text"], encoding="utf-8")
    paths["html"].write_text(payload["html"], encoding="utf-8")
    return paths
