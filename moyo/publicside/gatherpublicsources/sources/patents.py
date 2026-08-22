"""Patent search via USPTO PatentsView (optional key) or Google Patents."""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional
from urllib.parse import quote_plus

from moyo.publicside.gatherpublicsources.schema import PublicSource, SourceType
from moyo.publicside.gatherpublicsources.sources.http import (
    env,
    get_json,
    optional_http_url,
    parse_datetime,
)
from shared_utils import generate_id

logger = logging.getLogger(__name__)

PATENTSVIEW_URL = "https://search.patentsview.org/api/v1/patent/"
GOOGLE_PATENTS_XHR = "https://patents.google.com/xhr/query"
GOOGLE_PATENT_PAGE = "https://patents.google.com/patent/"


def _source(
    *,
    patent_id: str,
    title: str,
    abstract: str,
    url: Optional[str],
    published: Any,
    office: str,
    extra: Optional[dict[str, Any]] = None,
) -> PublicSource:
    patent_id = (patent_id or "").strip() or generate_id("patent")
    title = (title or "").strip() or f"Patent {patent_id}"
    return PublicSource(
        id=generate_id(f"patent_{patent_id}"),
        title=title,
        content=(abstract or title).strip(),
        source_type=SourceType.PATENT,
        url=optional_http_url(url) or f"{GOOGLE_PATENT_PAGE}{patent_id}",
        published_date=parse_datetime(published),
        organization=(extra or {}).get("assignee"),
        metadata={
            "patent_id": patent_id,
            "office": office,
            **(extra or {}),
        },
        tags=["patent", office.lower()],
    )


def _patentsview_query(query: str, max_results: int) -> dict[str, Any]:
    return {
        "q": {
            "_or": [
                {"patent_title": {"_text_any": query}},
                {"patent_abstract": {"_text_any": query}},
            ]
        },
        "f": [
            "patent_id",
            "patent_title",
            "patent_abstract",
            "patent_date",
            "assignees",
        ],
        "o": {"size": max(1, min(int(max_results), 100))},
    }


def _parse_patentsview(data: Any, max_results: int) -> List[PublicSource]:
    patents = []
    if not isinstance(data, dict):
        return patents
    rows = data.get("patents") or data.get("results") or []
    if not isinstance(rows, list):
        return patents
    for item in rows[:max_results]:
        if not isinstance(item, dict):
            continue
        patent_id = str(item.get("patent_id") or item.get("patent_number") or "")
        if not patent_id:
            continue
        assignees = item.get("assignees") or []
        assignee = None
        if isinstance(assignees, list) and assignees:
            first = assignees[0] if isinstance(assignees[0], dict) else {}
            assignee = first.get("assignee_organization") or first.get("assignee_name")
        patents.append(
            _source(
                patent_id=patent_id,
                title=str(item.get("patent_title") or ""),
                abstract=str(item.get("patent_abstract") or ""),
                url=f"{GOOGLE_PATENT_PAGE}{patent_id}",
                published=item.get("patent_date"),
                office="USPTO",
                extra={"assignee": assignee, "provider": "patentsview"},
            )
        )
    return patents


def _flatten_google_cluster(cluster: Any) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    if isinstance(cluster, dict):
        patent = cluster.get("patent") if isinstance(cluster.get("patent"), dict) else cluster
        if isinstance(patent, dict) and (
            patent.get("publication_number") or patent.get("patent_number")
        ):
            rows.append(patent)
        return rows
    if isinstance(cluster, list):
        for item in cluster:
            rows.extend(_flatten_google_cluster(item))
    return rows


def _parse_google_patents(data: Any, max_results: int) -> List[PublicSource]:
    patents = []
    if not isinstance(data, dict):
        return patents
    results = data.get("results") or {}
    clusters = results.get("cluster") if isinstance(results, dict) else data.get("cluster")
    rows: List[dict[str, Any]] = []
    if isinstance(clusters, list):
        for cluster in clusters:
            rows.extend(_flatten_google_cluster(cluster))
    elif isinstance(results, list):
        for item in results:
            rows.extend(_flatten_google_cluster(item))
    for patent in rows[:max_results]:
        patent_id = str(
            patent.get("publication_number")
            or patent.get("patent_number")
            or patent.get("id")
            or ""
        )
        if not patent_id:
            continue
        patents.append(
            _source(
                patent_id=patent_id,
                title=str(patent.get("title") or ""),
                abstract=str(patent.get("snippet") or patent.get("abstract") or ""),
                url=str(patent.get("url") or f"{GOOGLE_PATENT_PAGE}{patent_id}"),
                published=patent.get("publication_date") or patent.get("filing_date"),
                office="Google Patents",
                extra={"provider": "google_patents"},
            )
        )
    return patents


async def search_patents(
    query: str,
    max_results: int = 100,
    offices: List[str] | None = None,
) -> List[PublicSource]:
    """Search public patent records.

    Uses USPTO PatentsView when ``PATENTSVIEW_API_KEY`` is set, otherwise
    Google Patents' public xhr endpoint. ``offices`` is accepted for call-site
    compatibility; PatentsView is USPTO-centric.
    """
    del offices  # real adapter is USPTO / Google Patents, not EPO/JPO HTML
    query = (query or "").strip()
    if not query:
        return []
    max_results = max(1, min(int(max_results), 100))

    api_key = env("PATENTSVIEW_API_KEY")
    if api_key:
        payload = _patentsview_query(query, max_results)
        data = await get_json(
            PATENTSVIEW_URL,
            params={
                "q": json.dumps(payload["q"]),
                "f": json.dumps(payload["f"]),
                "o": json.dumps(payload["o"]),
            },
            headers={"X-Api-Key": api_key},
        )
        found = _parse_patentsview(data, max_results)
        if found:
            return found
        logger.info("PatentsView returned no rows; falling back to Google Patents")

    data = await get_json(
        GOOGLE_PATENTS_XHR,
        params={"url": f"q={quote_plus(query)}&num={max_results}"},
    )
    return _parse_google_patents(data, max_results)
