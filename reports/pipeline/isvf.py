"""Parse the Idea Security Verification Framework (ISVF) control catalog.

Remediation / investigation steps in the Basis Report are sourced from the
ISVF control catalog (``controls/control-catalog.md``) so the guidance stays in
sync with the framework rather than being hand-copied into the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ISVFControl:
    code: str  # e.g. "ISVF-EXF-01"
    family_code: str  # e.g. "ISVF-EXF"
    family_name: str  # e.g. "Exfiltration Controls"
    title: str
    mitigations: list[str] = field(default_factory=list)  # ISVF.M0013, ...
    summary: str = ""
    statement: str = ""
    evidence: list[str] = field(default_factory=list)


# Families most relevant to a MOYO exposure assessment (semantic leakage /
# cross-domain inference from multi-model public retrieval), in report order.
# Each maps to a short "why this applies" rationale rendered in the Basis Report.
DEFAULT_FAMILY_RATIONALE: dict[str, str] = {
    "ISVF-EXF": (
        "The assessment shows protected meaning is reconstructable from model "
        "outputs; exfiltration controls detect and contain that leakage."
    ),
    "ISVF-INF": (
        "Multiple models combined public fragments into higher-value "
        "conclusions; domain-reach controls test and gate that inference."
    ),
    "ISVF-CTX": (
        "Retrieval and context assembly widened what became reachable; "
        "least-privilege retrieval and session budgets constrain it."
    ),
    "ISVF-CLD": (
        "Findings were produced by consumer and vendor-hosted models; cloud "
        "prompting governance decides what may reach those planes."
    ),
    "ISVF-ADV": (
        "Reachability was demonstrated by adversarial prompting; recurring "
        "red-team testing keeps that risk measured over time."
    ),
    "ISVF-MON": (
        "Exposure drifts as models and prompts change; monitoring and drift "
        "detection catch regressions after changes."
    ),
    "ISVF-DATA": (
        "Sensitive meaning entered model-reachable space; pre-ingestion "
        "classification and labeling reduce that surface upstream."
    ),
    "ISVF-ASR": (
        "Findings should be packaged as auditable evidence of boundary "
        "claims, controls, and tests."
    ),
}

# Default control selection: family -> which controls (by code suffix) to
# surface, in priority order. Kept small so the remediation section is
# actionable rather than a full catalog dump.
DEFAULT_SELECTION: list[tuple[str, list[str]]] = [
    ("ISVF-EXF", ["ISVF-EXF-01", "ISVF-EXF-03"]),
    ("ISVF-INF", ["ISVF-INF-02", "ISVF-INF-01"]),
    ("ISVF-CTX", ["ISVF-CTX-01"]),
    ("ISVF-CLD", ["ISVF-CLD-01"]),
    ("ISVF-ADV", ["ISVF-ADV-01"]),
    ("ISVF-MON", ["ISVF-MON-02"]),
    ("ISVF-DATA", ["ISVF-DATA-02"]),
    ("ISVF-ASR", ["ISVF-ASR-01"]),
]

_FAMILY_TABLE_RE = re.compile(r"^\|\s*(ISVF-[A-Z]+)\s*\|\s*([^|]+?)\s*\|", re.M)
_CONTROL_HEADING_RE = re.compile(r"^###\s+(ISVF-[A-Z]+-\d+)\s+[—-]\s+(.+?)\s*$", re.M)
_MITIGATION_RE = re.compile(r"ISVF\.M\d{4}")


def _clean(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _family_of(code: str) -> str:
    # ISVF-EXF-01 -> ISVF-EXF
    parts = code.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else code


def load_isvf_controls(catalog_path: Path) -> list[ISVFControl]:
    """Parse ``control-catalog.md`` into structured controls.

    Returns an empty list if the catalog is missing so the pipeline degrades
    gracefully (Basis Report still renders, remediation section notes absence).
    """
    if not catalog_path.exists():
        return []
    text = catalog_path.read_text(encoding="utf-8")

    family_names = {
        code: _clean(name) for code, name in _FAMILY_TABLE_RE.findall(text)
    }

    controls: list[ISVFControl] = []
    matches = list(_CONTROL_HEADING_RE.finditer(text))
    for i, m in enumerate(matches):
        code = m.group(1).strip()
        title = _clean(m.group(2))
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]

        summary = ""
        sm = re.search(r"\*\*Summary:\*\*\s*(.+?)(?:\n\n|\n\*\*|\Z)", body, re.S)
        if sm:
            summary = _clean(sm.group(1))

        statement = ""
        st = re.search(
            r"\*\*Control Statement:\*\*\s*(.+?)(?:\n\n|\n\*\*|\Z)", body, re.S
        )
        if st:
            statement = _clean(st.group(1))

        evidence: list[str] = []
        ev = re.search(
            r"\*\*Evidence Requirements:\*\*\s*\n(.+?)(?:\n\n|\n<!--|\nSee |\Z)",
            body,
            re.S,
        )
        if ev:
            for line in ev.group(1).splitlines():
                s = line.strip()
                if s.startswith(("-", "*")):
                    evidence.append(_clean(s.lstrip("-* ").strip()))

        controls.append(
            ISVFControl(
                code=code,
                family_code=_family_of(code),
                family_name=family_names.get(_family_of(code), _family_of(code)),
                title=title,
                mitigations=_MITIGATION_RE.findall(body),
                summary=summary,
                statement=statement,
                evidence=evidence,
            )
        )
    return controls


def select_remediation(
    controls: list[ISVFControl],
    *,
    selection: list[tuple[str, list[str]]] | None = None,
) -> list[dict]:
    """Pick an actionable, ordered remediation set grouped by ISVF family.

    Each item: family_code, family_name, why (rationale), and controls (each
    with code, title, summary, statement, and investigation_steps derived from
    the control's evidence requirements).
    """
    if not controls:
        return []
    selection = selection or DEFAULT_SELECTION
    by_code = {c.code: c for c in controls}

    out: list[dict] = []
    for family_code, codes in selection:
        picked = [by_code[c] for c in codes if c in by_code]
        if not picked:
            continue
        out.append(
            {
                "family_code": family_code,
                "family_name": picked[0].family_name,
                "why": DEFAULT_FAMILY_RATIONALE.get(family_code, ""),
                "controls": [
                    {
                        "code": c.code,
                        "title": c.title,
                        "summary": c.summary,
                        "statement": c.statement,
                        "mitigations": c.mitigations,
                        "investigation_steps": c.evidence[:3],
                    }
                    for c in picked
                ],
            }
        )
    return out
