"""Public vulnerability / advisory search (NVD + GitHub Advisories).

TODO: Do not add GitHub dorks, paste-site scrapers, or credential harvesting.
Those were placeholder/hypothetical endpoints and are intentionally omitted.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from moyo.publicside.gatherpublicsources.schema import PublicSource, SourceType
from moyo.publicside.gatherpublicsources.sources.http import (
    env,
    get_json,
    optional_http_url,
    parse_datetime,
)
from shared_utils import generate_id

logger = logging.getLogger(__name__)

NVD_CVE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
GITHUB_ADVISORIES_URL = "https://api.github.com/advisories"


def _nvd_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    api_key = env("NVD_API_KEY")
    if api_key:
        headers["apiKey"] = api_key
    return headers


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = env("GITHUB_TOKEN") or env("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _cve_source(item: dict[str, Any]) -> Optional[PublicSource]:
    cve = item.get("cve") if isinstance(item.get("cve"), dict) else item
    cve_id = str(cve.get("id") or "").strip()
    if not cve_id:
        return None
    descriptions = cve.get("descriptions") if isinstance(cve.get("descriptions"), list) else []
    english = next(
        (d.get("value") for d in descriptions if isinstance(d, dict) and d.get("lang") == "en"),
        None,
    )
    if not english and descriptions and isinstance(descriptions[0], dict):
        english = descriptions[0].get("value")
    published = cve.get("published")
    refs = cve.get("references") if isinstance(cve.get("references"), list) else []
    url = None
    for ref in refs:
        if isinstance(ref, dict):
            url = optional_http_url(ref.get("url"))
            if url:
                break
    url = url or f"https://nvd.nist.gov/vuln/detail/{cve_id}"
    metrics = cve.get("metrics") if isinstance(cve.get("metrics"), dict) else {}
    severity = None
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        rows = metrics.get(key)
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            data = rows[0].get("cvssData") if isinstance(rows[0].get("cvssData"), dict) else {}
            severity = data.get("baseSeverity") or rows[0].get("baseSeverity")
            break
    return PublicSource(
        id=generate_id(cve_id),
        title=cve_id,
        content=str(english or cve_id),
        source_type=SourceType.LEAKED_CODE,
        url=url,
        published_date=parse_datetime(published),
        organization="NVD",
        metadata={
            "cve_id": cve_id,
            "severity": severity,
            "provider": "nvd",
        },
        tags=["advisory", "cve", "nvd"] + ([str(severity).lower()] if severity else []),
    )


def _ghsa_source(item: dict[str, Any]) -> Optional[PublicSource]:
    ghsa_id = str(item.get("ghsa_id") or item.get("id") or "").strip()
    summary = str(item.get("summary") or "").strip()
    if not ghsa_id and not summary:
        return None
    title = ghsa_id or summary
    if ghsa_id and summary:
        title = f"{ghsa_id}: {summary}"
    return PublicSource(
        id=generate_id(ghsa_id or title),
        title=title[:200],
        content=str(item.get("description") or summary or title),
        source_type=SourceType.LEAKED_CODE,
        url=optional_http_url(item.get("html_url")),
        published_date=parse_datetime(item.get("published_at") or item.get("updated_at")),
        organization="GitHub Advisory Database",
        metadata={
            "ghsa_id": ghsa_id,
            "severity": item.get("severity"),
            "cve_id": (item.get("cve_id") or (item.get("identifiers") or [None])[0]),
            "provider": "github_advisories",
        },
        tags=["advisory", "ghsa"] + ([str(item.get("severity")).lower()] if item.get("severity") else []),
    )


async def search_leaked_code(
    query: str,
    max_results: int = 100,
    sources: List[str] | None = None,
) -> List[PublicSource]:
    """Search public vulnerability databases for the topic.

    Replaces the old placeholder ``example.com/leaks`` adapter. This does
    **not** scrape paste sites or run GitHub credential dorks.
    """
    query = (query or "").strip()
    if not query:
        return []
    max_results = max(1, min(int(max_results), 100))
    wanted = {(s or "").strip().lower() for s in (sources or ["nvd", "github_advisories"])}
    # Old crawler passed github_dorks / security_forums — map to public advisories.
    if "github_dorks" in wanted or "security_forums" in wanted:
        wanted.update({"nvd", "github_advisories"})
    per_source = max(1, max_results // 2)
    results: List[PublicSource] = []

    if not wanted or "nvd" in wanted:
        data = await get_json(
            NVD_CVE_URL,
            params={
                "keywordSearch": query,
                "resultsPerPage": str(min(per_source, 20)),
            },
            headers=_nvd_headers(),
        )
        vulns = data.get("vulnerabilities") if isinstance(data, dict) else None
        if isinstance(vulns, list):
            for item in vulns[:per_source]:
                if isinstance(item, dict):
                    source = _cve_source(item)
                    if source:
                        results.append(source)

    if "github_advisories" in wanted or "github" in wanted:
        data = await get_json(
            GITHUB_ADVISORIES_URL,
            params={"query": query, "per_page": str(per_source)},
            headers=_github_headers(),
        )
        rows = data if isinstance(data, list) else []
        for item in rows[:per_source]:
            if isinstance(item, dict):
                source = _ghsa_source(item)
                if source:
                    results.append(source)

    skipped = wanted - {
        "nvd",
        "github_advisories",
        "github",
        "github_dorks",
        "security_forums",
        "",
    }
    if skipped:
        logger.info(
            "Skipping leak sources that are not implemented: %s",
            ", ".join(sorted(skipped)),
        )
    return results[:max_results]
