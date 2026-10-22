"""Report voice by scan audience.

Storefront voices are organization, personal, competitive, security, and
opposition. Each MoyoMap type has its own voice as well: corporate
investigations, due diligence, red teaming, incident, and free form
(``other``), plus the personal and opposition voices they already share.
The generic organization builder is unchanged. Other voices rename
disclosure bands; every extracted claim is retained.
"""

from __future__ import annotations

import re
from typing import Any

ORGANIZATION = "organization"
OPPOSITION = "opposition"
PERSONAL = "personal"
COMPETITIVE = "competitive"
SECURITY = "security"
# Map-type voices. Ids match MoyoMap categories. Storefront voices stay above.
CORPORATE_INVESTIGATIONS = "corporate_investigations"
DUE_DILIGENCE = "due_diligence"
RED_TEAMING = "red_teaming"
INCIDENT = "incident"
OTHER = "other"
POLITICAL = "political"
HISTORY = "history"
ECONOMIC = "economic"
TECH = "tech"
NEWS = "news"

AUDIENCE_CHOICES = (
    ORGANIZATION,
    PERSONAL,
    COMPETITIVE,
    SECURITY,
    OPPOSITION,
    POLITICAL,
    HISTORY,
    ECONOMIC,
    TECH,
    NEWS,
    CORPORATE_INVESTIGATIONS,
    DUE_DILIGENCE,
    RED_TEAMING,
    INCIDENT,
    OTHER,
)

# Company and city, not school. Org-shaped map voices share this with the storefront.
SUBJECT_COMPANY_CITY = frozenset(
    {COMPETITIVE, SECURITY, CORPORATE_INVESTIGATIONS, DUE_DILIGENCE, RED_TEAMING}
)

# (priority label, second band). Sensitivity remap shared by these voices.
_BAND_REMAP: dict[str, tuple[str, str]] = {
    COMPETITIVE: ("Commercially sensitive", "Potentially strategic"),
    SECURITY: ("Security relevant", "Material"),
    CORPORATE_INVESTIGATIONS: ("Material", "Follow up"),
    DUE_DILIGENCE: ("Decision-critical", "Qualify"),
    RED_TEAMING: ("Exposed", "Harden"),
    INCIDENT: ("Pivotal", "Timeline"),
    OTHER: ("Significant", "Notable"),
    HISTORY: ("Significant", "Notable"),
    ECONOMIC: ("Significant", "Notable"),
    TECH: ("Significant", "Notable"),
    NEWS: ("Significant", "Notable"),
}

_OPPO_COVER_RE = re.compile(
    r"compile opposition research from public sources on (?P<name>.+?)"
    r"(?: as a candidate in (?P<race>.+?))?"
    r"(?=\.\s*(?:report documented|disambiguate)|$)",
    re.I | re.S,
)
_PERSONAL_COVER_RE = re.compile(
    r"what personal, biographical, and lesser-known public information "
    r"do ai systems associate with (?P<name>.+?)\?",
    re.I | re.S,
)
_ORG_COVER_RE = re.compile(
    r"what do ai systems already know about (?P<name>.+?) from public information\?",
    re.I | re.S,
)
_NAME_DETAIL_RE = re.compile(r"^(?P<name>.+?)\s+\((?P<detail>[^)]+)\)\s*$")
_TOPIC_PREFIX_RE = re.compile(r"^#\s*topic exploration:\s*", re.I)

_BASIC_CLASS_TO_BIN = {
    "Security relevant": "security_relevant",
    "Unexpected": "unexpected",
    "Interesting": "interesting",
    "Expected": "expected",
}

_OPPO_CLASS_TO_BIN = {
    "Damaging": "damaging",
    "Potentially damaging": "potentially_damaging",
    "Unexpected": "unexpected",
    "Interesting": "interesting",
    "Expected": "expected",
}

_PERSONAL_CLASS_TO_BIN = {
    "Sensitive": "sensitive",
    "Unexpected": "unexpected",
    "Interesting": "interesting",
    "Expected": "expected",
    "Security relevant": "sensitive",
}

_COMPETITIVE_CLASS_TO_BIN = {
    "Commercially sensitive": "commercially_sensitive",
    "Potentially strategic": "potentially_strategic",
    "Unexpected": "unexpected",
    "Interesting": "interesting",
    "Expected": "expected",
}

_SECURITY_CLASS_TO_BIN = {
    "Security relevant": "security_relevant",
    "Material": "material",
    "Unexpected": "unexpected",
    "Interesting": "interesting",
    "Expected": "expected",
}

_CORPORATE_CLASS_TO_BIN = {
    "Material": "corporate_material",
    "Follow up": "corporate_follow_up",
    "Unexpected": "unexpected",
    "Interesting": "interesting",
    "Expected": "expected",
}

_DUE_DILIGENCE_CLASS_TO_BIN = {
    "Decision-critical": "decision_critical",
    "Qualify": "qualify",
    "Unexpected": "unexpected",
    "Interesting": "interesting",
    "Expected": "expected",
}

_RED_TEAM_CLASS_TO_BIN = {
    "Exposed": "exposed",
    "Harden": "harden",
    "Unexpected": "unexpected",
    "Interesting": "interesting",
    "Expected": "expected",
}

_INCIDENT_CLASS_TO_BIN = {
    "Pivotal": "pivotal",
    "Timeline": "timeline",
    "Unexpected": "unexpected",
    "Interesting": "interesting",
    "Expected": "expected",
}

_OTHER_CLASS_TO_BIN = {
    "Significant": "significant",
    "Notable": "notable",
    "Unexpected": "unexpected",
    "Interesting": "interesting",
    "Expected": "expected",
}

CHART_ORDER = {
    ORGANIZATION: ("security_relevant", "unexpected", "interesting", "expected"),
    OPPOSITION: ("damaging", "potentially_damaging", "unexpected", "interesting"),
    PERSONAL: ("sensitive", "unexpected", "interesting", "expected"),
    COMPETITIVE: (
        "commercially_sensitive",
        "potentially_strategic",
        "unexpected",
        "interesting",
    ),
    SECURITY: ("security_relevant", "material", "unexpected", "interesting"),
    CORPORATE_INVESTIGATIONS: (
        "corporate_material",
        "corporate_follow_up",
        "unexpected",
        "interesting",
    ),
    DUE_DILIGENCE: ("decision_critical", "qualify", "unexpected", "interesting"),
    RED_TEAMING: ("exposed", "harden", "unexpected", "interesting"),
    INCIDENT: ("pivotal", "timeline", "unexpected", "interesting", "expected"),
    OTHER: ("significant", "notable", "unexpected", "interesting", "expected"),
}
CHART_ORDER[POLITICAL] = CHART_ORDER[OPPOSITION]

# Lower sorts first in inventories and on the one-pager.
CLASS_RANK = {
    ORGANIZATION: {
        "Security relevant": 0,
        "Unexpected": 1,
        "Interesting": 2,
        "Expected": 3,
    },
    OPPOSITION: {
        "Damaging": 0,
        "Potentially damaging": 1,
        "Unexpected": 2,
        "Interesting": 3,
        "Expected": 9,
    },
    PERSONAL: {
        "Expected": 0,
        "Interesting": 1,
        "Unexpected": 2,
        "Sensitive": 3,
    },
    COMPETITIVE: {
        "Commercially sensitive": 0,
        "Potentially strategic": 1,
        "Unexpected": 2,
        "Interesting": 3,
        "Expected": 9,
    },
    SECURITY: {
        "Security relevant": 0,
        "Material": 1,
        "Unexpected": 2,
        "Interesting": 3,
        "Expected": 9,
    },
    CORPORATE_INVESTIGATIONS: {
        "Material": 0,
        "Follow up": 1,
        "Unexpected": 2,
        "Interesting": 3,
        "Expected": 9,
    },
    DUE_DILIGENCE: {
        "Decision-critical": 0,
        "Qualify": 1,
        "Unexpected": 2,
        "Interesting": 3,
        "Expected": 9,
    },
    RED_TEAMING: {
        "Exposed": 0,
        "Harden": 1,
        "Unexpected": 2,
        "Interesting": 3,
        "Expected": 9,
    },
    INCIDENT: {
        "Pivotal": 0,
        "Timeline": 1,
        "Unexpected": 2,
        "Interesting": 3,
        "Expected": 9,
    },
    OTHER: {
        "Significant": 0,
        "Notable": 1,
        "Unexpected": 2,
        "Interesting": 3,
        "Expected": 9,
    },
}
CLASS_RANK[POLITICAL] = CLASS_RANK[OPPOSITION]
for _catalog_voice in (HISTORY, ECONOMIC, TECH, NEWS):
    CHART_ORDER[_catalog_voice] = CHART_ORDER[OTHER]
    CLASS_RANK[_catalog_voice] = CLASS_RANK[OTHER]


def opposition_shaped(audience: str) -> bool:
    """Opposition research and political-race maps share disclosure bands."""
    return normalize_audience(audience) in (OPPOSITION, POLITICAL)


def normalize_audience(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if text in AUDIENCE_CHOICES:
        return text
    return ORGANIZATION


def lead_labels(audience: str = ORGANIZATION) -> tuple[str, str] | None:
    """Priority band and the band under it, for voices that remap sensitivity."""
    return _BAND_REMAP.get(normalize_audience(audience))


def _prompt_body(raw: Any) -> str:
    text = str(raw or "").strip()
    return _TOPIC_PREFIX_RE.sub("", text, count=1).strip()


def infer_audience_from_text(raw: Any) -> str | None:
    """Scan audience when ``raw`` looks like a storefront retrieval wrap.

    Competitive and security share a wrap; this returns competitive for that
    text. Pick ``security`` explicitly on the CLI or GUI to use that voice.
    """
    text = _prompt_body(raw)
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("compile opposition research"):
        return OPPOSITION
    if lowered.startswith("map this political race"):
        return POLITICAL
    if lowered.startswith("research the historical record"):
        return HISTORY
    if lowered.startswith("research the economics of"):
        return ECONOMIC
    if lowered.startswith("research the technology and technical record"):
        return TECH
    if lowered.startswith("map what reporting establishes"):
        return NEWS
    if lowered.startswith("what personal, biographical"):
        return PERSONAL
    if lowered.startswith("compile commercial due diligence"):
        return DUE_DILIGENCE
    if lowered.startswith("map what public sources already show about the security posture"):
        return RED_TEAMING
    if lowered.startswith("reconstruct the public record of this incident"):
        return INCIDENT
    if lowered.startswith("research this topic from public information"):
        return OTHER
    if "unannounced or non-public product" in lowered or "not in official marketing" in lowered:
        return COMPETITIVE
    if lowered.startswith("what do ai systems already know"):
        return ORGANIZATION
    return None


def infer_cover_from_text(raw: Any) -> dict[str, str]:
    """Best-effort audience, cover name, and subject detail from a wrap."""
    text = _prompt_body(raw)
    out: dict[str, str] = {}
    audience = infer_audience_from_text(text)
    if audience:
        out["audience"] = audience
    if not text:
        return out

    oppo = _OPPO_COVER_RE.search(text)
    if oppo:
        name = (oppo.group("name") or "").strip()
        race = (oppo.group("race") or "").strip()
        if name and race:
            out["display_topic"] = f"{name} — {race}"
        elif name:
            out["display_topic"] = name
        return out

    personal = _PERSONAL_COVER_RE.search(text)
    if personal:
        name = (personal.group("name") or "").strip()
        if name:
            out["display_topic"] = name
        return out

    org = _ORG_COVER_RE.search(text)
    if org:
        blob = (org.group("name") or "").strip()
        detail = _NAME_DETAIL_RE.match(blob)
        if detail:
            out["display_topic"] = detail.group("name").strip()
            out["subject_detail"] = detail.group("detail").strip()
        elif blob:
            out["display_topic"] = blob
    return out


def resolve_audience(
    *,
    explicit: Any = None,
    config: Any = None,
    prior: Any = None,
    texts: list[Any] | None = None,
) -> str:
    """CLI / config beat wrap inference; wrap beats a stale prior report."""
    for raw in (explicit, config):
        if str(raw or "").strip():
            return normalize_audience(raw)
    for raw in texts or []:
        inferred = infer_audience_from_text(raw)
        if inferred:
            return inferred
    if str(prior or "").strip():
        return normalize_audience(prior)
    return ORGANIZATION


def resolve_cover_field(
    *,
    key: str,
    explicit: Any = None,
    config: Any = None,
    prior: Any = None,
    texts: list[Any] | None = None,
) -> str:
    """Resolve ``display_topic`` or ``subject_detail`` the same way as audience."""
    for raw in (explicit, config):
        text = str(raw or "").strip()
        if text:
            return text
    for raw in texts or []:
        inferred = infer_cover_from_text(raw).get(key, "").strip()
        if inferred:
            return inferred
    return str(prior or "").strip()


def base_disclosure_class(finding: dict) -> str:
    """Organization-builder bucket. Other voices remap this label."""
    sens = int(finding.get("sensitivity") or 0)
    novelty = int(finding.get("novelty") or 0)
    corr = int(finding.get("corroboration") or 1)
    if sens >= 4:
        return "Security relevant"
    if novelty >= 4 or (novelty >= 3 and corr >= 2):
        return "Unexpected"
    if novelty >= 2 or sens >= 3:
        return "Interesting"
    if corr >= 2 and sens >= 2:
        return "Unexpected"
    return "Expected"


def disclosure_class(finding: dict, audience: str = ORGANIZATION) -> str:
    audience = normalize_audience(audience)
    base = base_disclosure_class(finding)
    sens = int(finding.get("sensitivity") or 0)
    if opposition_shaped(audience):
        if base == "Security relevant" or sens >= 4:
            return "Damaging"
        if base == "Unexpected":
            return "Unexpected"
        if sens >= 3:
            return "Potentially damaging"
        if base == "Interesting":
            return "Interesting"
        return "Expected"
    if audience == PERSONAL:
        if base == "Security relevant":
            return "Sensitive"
        return base
    pair = _BAND_REMAP.get(audience)
    if pair:
        top, mid = pair
        if base == "Security relevant" or sens >= 4:
            return top
        if base == "Unexpected":
            return "Unexpected"
        if sens >= 3:
            return mid
        if base == "Interesting":
            return "Interesting"
        return "Expected"
    return base


def disclosure_bin(finding: dict, audience: str = ORGANIZATION) -> str:
    audience = normalize_audience(audience)
    label = disclosure_class(finding, audience)
    table = {
        OPPOSITION: _OPPO_CLASS_TO_BIN,
        POLITICAL: _OPPO_CLASS_TO_BIN,
        PERSONAL: _PERSONAL_CLASS_TO_BIN,
        COMPETITIVE: _COMPETITIVE_CLASS_TO_BIN,
        SECURITY: _SECURITY_CLASS_TO_BIN,
        CORPORATE_INVESTIGATIONS: _CORPORATE_CLASS_TO_BIN,
        DUE_DILIGENCE: _DUE_DILIGENCE_CLASS_TO_BIN,
        RED_TEAMING: _RED_TEAM_CLASS_TO_BIN,
        INCIDENT: _INCIDENT_CLASS_TO_BIN,
        OTHER: _OTHER_CLASS_TO_BIN,
        HISTORY: _OTHER_CLASS_TO_BIN,
        ECONOMIC: _OTHER_CLASS_TO_BIN,
        TECH: _OTHER_CLASS_TO_BIN,
        NEWS: _OTHER_CLASS_TO_BIN,
    }.get(audience, _BASIC_CLASS_TO_BIN)
    return table.get(label, "expected")


def chart_order(audience: str = ORGANIZATION) -> tuple[str, ...]:
    return CHART_ORDER[normalize_audience(audience)]


def retain_finding(finding: dict, audience: str = ORGANIZATION) -> bool:
    """Keep every extracted claim. Audience still remaps disclosure labels."""
    del finding, audience
    return True


def sort_key(finding: dict, audience: str = ORGANIZATION) -> tuple:
    audience = normalize_audience(audience)
    label = disclosure_class(finding, audience)
    rank = CLASS_RANK[audience].get(label, 9)
    return (
        rank,
        -int(finding.get("sensitivity") or 0),
        -int(finding.get("specificity") or 0),
        -int(finding.get("novelty") or 0),
        str(finding.get("claim_id") or ""),
    )


def profile(audience: str = ORGANIZATION) -> dict[str, str]:
    audience = normalize_audience(audience)
    if audience == OPPOSITION:
        return {
            "audience": OPPOSITION,
            "kicker": "Opposition research",
            "basis_kicker": "Opposition research",
            "eyebrow": "Opposition research",
            "priority_label": "Damaging",
            "headline": "What models assemble from public records",
        }
    if audience == POLITICAL:
        return {
            "audience": POLITICAL,
            "kicker": "Political race",
            "basis_kicker": "Political race",
            "eyebrow": "Political race",
            "priority_label": "Damaging",
            "headline": "What the public record shows about this race",
        }
    if audience == PERSONAL:
        return {
            "audience": PERSONAL,
            "kicker": "Personal exposure",
            "basis_kicker": "Personal exposure",
            "eyebrow": "Personal exposure",
            "priority_label": "Sensitive",
            "headline": "What models associate with this person",
        }
    if audience == COMPETITIVE:
        return {
            "audience": COMPETITIVE,
            "kicker": "Competitive intelligence",
            "basis_kicker": "Competitive intelligence",
            "eyebrow": "Competitive intelligence",
            "priority_label": "Commercially sensitive",
            "headline": "What models already know about this competitor",
        }
    if audience == SECURITY:
        return {
            "audience": SECURITY,
            "kicker": "Security exposure",
            "basis_kicker": "Security exposure",
            "eyebrow": "Security exposure",
            "priority_label": "Security relevant",
            "headline": "What an outsider can already reconstruct",
        }
    if audience == CORPORATE_INVESTIGATIONS:
        return {
            "audience": CORPORATE_INVESTIGATIONS,
            "kicker": "Corporate investigation",
            "basis_kicker": "Corporate investigation",
            "eyebrow": "Corporate investigation",
            "priority_label": "Material",
            "headline": "What the public record shows about this organization",
        }
    if audience == DUE_DILIGENCE:
        return {
            "audience": DUE_DILIGENCE,
            "kicker": "Due diligence",
            "basis_kicker": "Due diligence",
            "eyebrow": "Due diligence",
            "priority_label": "Decision-critical",
            "headline": "What a buyer can already establish",
        }
    if audience == RED_TEAMING:
        return {
            "audience": RED_TEAMING,
            "kicker": "Red teaming",
            "basis_kicker": "Red teaming",
            "eyebrow": "Red teaming",
            "priority_label": "Exposed",
            "headline": "What public sources already show about this posture",
        }
    if audience == INCIDENT:
        return {
            "audience": INCIDENT,
            "kicker": "Incident record",
            "basis_kicker": "Incident record",
            "eyebrow": "Incident record",
            "priority_label": "Pivotal",
            "headline": "What the public record establishes",
        }
    if audience == HISTORY:
        return {
            "audience": HISTORY,
            "kicker": "History",
            "basis_kicker": "History",
            "eyebrow": "History",
            "priority_label": "Significant",
            "headline": "What the historical record supports",
        }
    if audience == ECONOMIC:
        return {
            "audience": ECONOMIC,
            "kicker": "Economic",
            "basis_kicker": "Economic",
            "eyebrow": "Economic",
            "priority_label": "Significant",
            "headline": "What the economic record shows",
        }
    if audience == TECH:
        return {
            "audience": TECH,
            "kicker": "Tech",
            "basis_kicker": "Tech",
            "eyebrow": "Tech",
            "priority_label": "Significant",
            "headline": "What the technical record shows",
        }
    if audience == NEWS:
        return {
            "audience": NEWS,
            "kicker": "News",
            "basis_kicker": "News",
            "eyebrow": "News",
            "priority_label": "Significant",
            "headline": "What reporting establishes",
        }
    if audience == OTHER:
        return {
            "audience": OTHER,
            "kicker": "Research map",
            "basis_kicker": "Research map",
            "eyebrow": "Research map",
            "priority_label": "Significant",
            "headline": "What public sources show",
        }
    return {
        "audience": ORGANIZATION,
        "kicker": "Exposure assessment",
        "basis_kicker": "Basis report",
        "eyebrow": "Exposure snapshot",
        "priority_label": "High-sensitivity",
        "headline": "What AI Systems Reveal",
    }


def stock_headlines() -> set[str]:
    """Default cover titles; do not treat these as editorial overrides."""
    return {profile(name)["headline"] for name in AUDIENCE_CHOICES}


def priority_count(findings: list[dict], audience: str = ORGANIZATION) -> int:
    audience = normalize_audience(audience)
    if opposition_shaped(audience):
        return sum(1 for f in findings if disclosure_class(f, audience) == "Damaging")
    if audience == PERSONAL:
        return sum(1 for f in findings if disclosure_class(f, audience) == "Sensitive")
    pair = _BAND_REMAP.get(audience)
    if pair:
        top = pair[0]
        return sum(1 for f in findings if disclosure_class(f, audience) == top)
    return sum(1 for f in findings if int(f.get("sensitivity") or 0) >= 4)
