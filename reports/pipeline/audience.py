"""Report voice by storefront scan audience.

The generic organization builder is unchanged. Competitive intelligence and
security each keep Interesting and Unexpected, drop ordinary Expected facts,
and rename the top bands. Opposition and personal keep their own voices.
"""

from __future__ import annotations

from typing import Any

ORGANIZATION = "organization"
OPPOSITION = "opposition"
PERSONAL = "personal"
COMPETITIVE = "competitive"
SECURITY = "security"

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
}

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
}


def normalize_audience(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if text == OPPOSITION:
        return OPPOSITION
    if text == PERSONAL:
        return PERSONAL
    if text == COMPETITIVE:
        return COMPETITIVE
    if text == SECURITY:
        return SECURITY
    return ORGANIZATION


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
    if audience == OPPOSITION:
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
    if audience == COMPETITIVE:
        if base == "Security relevant" or sens >= 4:
            return "Commercially sensitive"
        if base == "Unexpected":
            return "Unexpected"
        if sens >= 3:
            return "Potentially strategic"
        if base == "Interesting":
            return "Interesting"
        return "Expected"
    if audience == SECURITY:
        if base == "Security relevant" or sens >= 4:
            return "Security relevant"
        if base == "Unexpected":
            return "Unexpected"
        if sens >= 3:
            return "Material"
        if base == "Interesting":
            return "Interesting"
        return "Expected"
    return base


def disclosure_bin(finding: dict, audience: str = ORGANIZATION) -> str:
    audience = normalize_audience(audience)
    label = disclosure_class(finding, audience)
    table = {
        OPPOSITION: _OPPO_CLASS_TO_BIN,
        PERSONAL: _PERSONAL_CLASS_TO_BIN,
        COMPETITIVE: _COMPETITIVE_CLASS_TO_BIN,
        SECURITY: _SECURITY_CLASS_TO_BIN,
    }.get(audience, _BASIC_CLASS_TO_BIN)
    return table.get(label, "expected")


def chart_order(audience: str = ORGANIZATION) -> tuple[str, ...]:
    return CHART_ORDER[normalize_audience(audience)]


def retain_finding(finding: dict, audience: str = ORGANIZATION) -> bool:
    """Opposition, competitive, and security reports omit ordinary public facts."""
    audience = normalize_audience(audience)
    if audience not in {OPPOSITION, COMPETITIVE, SECURITY}:
        return True
    if int(finding.get("sensitivity") or 0) <= 1:
        return False
    return disclosure_class(finding, audience) != "Expected"


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
    return {
        "audience": ORGANIZATION,
        "kicker": "Exposure assessment",
        "basis_kicker": "Basis report",
        "eyebrow": "Exposure snapshot",
        "priority_label": "High-sensitivity",
        "headline": "What AI Systems Reveal",
    }


def priority_count(findings: list[dict], audience: str = ORGANIZATION) -> int:
    audience = normalize_audience(audience)
    if audience == OPPOSITION:
        return sum(1 for f in findings if disclosure_class(f, audience) == "Damaging")
    if audience == PERSONAL:
        return sum(1 for f in findings if disclosure_class(f, audience) == "Sensitive")
    if audience == COMPETITIVE:
        return sum(
            1 for f in findings if disclosure_class(f, audience) == "Commercially sensitive"
        )
    if audience == SECURITY:
        return sum(1 for f in findings if disclosure_class(f, audience) == "Security relevant")
    return sum(1 for f in findings if int(f.get("sensitivity") or 0) >= 4)
