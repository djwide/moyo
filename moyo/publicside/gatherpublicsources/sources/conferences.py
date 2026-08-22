"""Conference / preprint search via arXiv and OpenAlex (no paid APIs)."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any, List, Optional

from moyo.publicside.gatherpublicsources.schema import PublicSource, SourceType
from moyo.publicside.gatherpublicsources.sources.http import (
    env,
    get_json,
    get_text,
    optional_http_url,
    parse_datetime,
    reconstruct_inverted_abstract,
)
from shared_utils import generate_id

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_WORKS = "https://api.openalex.org/works"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _text(el: Optional[ET.Element]) -> str:
    if el is None or el.text is None:
        return ""
    return " ".join(el.text.split())


def _parse_arxiv_atom(xml_text: str, max_results: int) -> List[PublicSource]:
    if not xml_text.strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("arXiv response was not valid Atom XML")
        return []
    results: List[PublicSource] = []
    for entry in root.findall("atom:entry", ATOM_NS)[:max_results]:
        title = _text(entry.find("atom:title", ATOM_NS))
        summary = _text(entry.find("atom:summary", ATOM_NS))
        published = _text(entry.find("atom:published", ATOM_NS))
        arxiv_id = _text(entry.find("atom:id", ATOM_NS))
        authors = [
            _text(name)
            for name in entry.findall("atom:author/atom:name", ATOM_NS)
            if _text(name)
        ]
        pdf_url = None
        html_url = optional_http_url(arxiv_id)
        for link in entry.findall("atom:link", ATOM_NS):
            href = link.attrib.get("href", "")
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = optional_http_url(href)
            elif link.attrib.get("rel") == "alternate":
                html_url = optional_http_url(href) or html_url
        if not title:
            continue
        results.append(
            PublicSource(
                id=generate_id(arxiv_id or title),
                title=title,
                content=summary or title,
                source_type=SourceType.CONFERENCE_TALK,
                url=html_url or pdf_url,
                published_date=parse_datetime(published),
                author=", ".join(authors) if authors else None,
                organization="arXiv",
                metadata={
                    "arxiv_id": arxiv_id,
                    "pdf_url": pdf_url,
                    "authors": authors,
                    "provider": "arxiv",
                },
                tags=["conference", "preprint", "arxiv"],
            )
        )
    return results


def _parse_openalex(data: Any, max_results: int) -> List[PublicSource]:
    results: List[PublicSource] = []
    rows = data.get("results") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return results
    for item in rows[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("display_name") or item.get("title") or "").strip()
        if not title:
            continue
        location = item.get("primary_location") if isinstance(item.get("primary_location"), dict) else {}
        source = location.get("source") if isinstance(location.get("source"), dict) else {}
        authorships = item.get("authorships") if isinstance(item.get("authorships"), list) else []
        authors = []
        for auth in authorships:
            author = auth.get("author") if isinstance(auth, dict) else None
            name = (author or {}).get("display_name") if isinstance(author, dict) else None
            if name:
                authors.append(str(name))
        abstract = reconstruct_inverted_abstract(item.get("abstract_inverted_index"))
        landing = optional_http_url(location.get("landing_page_url")) or optional_http_url(
            item.get("id")
        )
        venue = source.get("display_name")
        results.append(
            PublicSource(
                id=generate_id(str(item.get("id") or title)),
                title=title,
                content=abstract or title,
                source_type=SourceType.CONFERENCE_TALK,
                url=landing,
                published_date=parse_datetime(item.get("publication_date")),
                author=", ".join(authors) if authors else None,
                organization=str(venue) if venue else "OpenAlex",
                metadata={
                    "openalex_id": item.get("id"),
                    "venue": venue,
                    "type": item.get("type"),
                    "authors": authors,
                    "provider": "openalex",
                },
                tags=["conference", "openalex"],
            )
        )
    return results


async def search_conference_talks(
    query: str,
    max_results: int = 100,
    sources: List[str] | None = None,
) -> List[PublicSource]:
    """Search scholarly talks/papers.

    Default sources are arXiv (Atom API) and OpenAlex. IEEE Xplore, ACM DL,
    YouTube, and SlideShare are not public unauthenticated APIs — they are not
    called. Set ``OPENALEX_MAILTO`` for OpenAlex's polite pool.
    """
    query = (query or "").strip()
    if not query:
        return []
    max_results = max(1, min(int(max_results), 100))
    wanted = {(s or "").strip().lower() for s in (sources or ["arxiv", "openalex"])}
    # Map the old crawler defaults onto the open replacements.
    if "ieee" in wanted or "acm" in wanted:
        wanted.add("openalex")
    per_source = max(1, max_results // max(1, len(wanted) or 1))
    results: List[PublicSource] = []

    if not wanted or "arxiv" in wanted:
        xml_text = await get_text(
            ARXIV_API,
            params={
                "search_query": f'all:"{query}"',
                "start": "0",
                "max_results": str(per_source),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        results.extend(_parse_arxiv_atom(xml_text, per_source))

    if "openalex" in wanted:
        params: dict[str, str] = {
            "search": query,
            "per_page": str(per_source),
            "sort": "publication_date:desc",
        }
        mailto = env("OPENALEX_MAILTO")
        if mailto:
            params["mailto"] = mailto
        data = await get_json(OPENALEX_WORKS, params=params)
        results.extend(_parse_openalex(data, per_source))

    unused = wanted - {"arxiv", "openalex", "ieee", "acm", ""}
    if unused:
        logger.info(
            "Skipping conference sources without a public API: %s",
            ", ".join(sorted(unused)),
        )
    return results[:max_results]
