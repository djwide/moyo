"""Press / news search via the GDELT 2.0 DOC API (no key)."""

from __future__ import annotations

import logging
from typing import List, Optional
from urllib.parse import quote_plus

from moyo.publicside.gatherpublicsources.schema import PublicSource, SourceType
from moyo.publicside.gatherpublicsources.sources.http import (
    get_json,
    optional_http_url,
    parse_datetime,
)
from shared_utils import generate_id

logger = logging.getLogger(__name__)

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Wire-service / news domains commonly used for company announcements.
DEFAULT_DOMAINS = (
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
    "reuters.com",
    "apnews.com",
)

DOMAIN_ALIASES = {
    "prnewswire": "prnewswire.com",
    "businesswire": "businesswire.com",
    "globenewswire": "globenewswire.com",
    "reuters": "reuters.com",
    "ap": "apnews.com",
    "apnews": "apnews.com",
}


def _domains_from_sources(sources: Optional[List[str]]) -> List[str]:
    if not sources:
        return list(DEFAULT_DOMAINS)
    out: List[str] = []
    for raw in sources:
        key = (raw or "").strip().lower()
        if not key:
            continue
        domain = DOMAIN_ALIASES.get(key, key)
        if domain not in out:
            out.append(domain)
    return out or list(DEFAULT_DOMAINS)


async def search_press_releases(
    query: str,
    max_results: int = 100,
    sources: List[str] | None = None,
) -> List[PublicSource]:
    """Search recent articles on public newswire / press domains via GDELT.

    GDELT is an open research project (https://www.gdeltproject.org/). No API
    key. Domain filters default to PR Newswire, Business Wire, GlobeNewswire,
    Reuters, and AP.
    """
    query = (query or "").strip()
    if not query:
        return []
    max_results = max(1, min(int(max_results), 75))  # GDELT ArtList cap is 75
    domains = _domains_from_sources(sources)
    domain_clause = " ".join(f"domain:{d}" for d in domains)
    gdelt_query = f"({query}) ({domain_clause})"

    data = await get_json(
        GDELT_DOC_URL,
        params={
            "query": gdelt_query,
            "mode": "ArtList",
            "maxrecords": str(max_results),
            "format": "json",
            "sort": "DateDesc",
        },
    )
    articles = []
    if isinstance(data, dict):
        articles = data.get("articles") or []
    if not isinstance(articles, list):
        return []

    results: List[PublicSource] = []
    for item in articles[:max_results]:
        if not isinstance(item, dict):
            continue
        url = optional_http_url(item.get("url"))
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        domain = str(item.get("domain") or "")
        results.append(
            PublicSource(
                id=generate_id(title),
                title=title,
                content=title,
                source_type=SourceType.PRESS_RELEASE,
                url=url,
                published_date=parse_datetime(item.get("seendate")),
                organization=domain or None,
                metadata={
                    "domain": domain,
                    "language": item.get("language"),
                    "sourcecountry": item.get("sourcecountry"),
                    "provider": "gdelt",
                    "query": quote_plus(query),
                },
                tags=["press", "news", "gdelt"],
            )
        )
    return results
