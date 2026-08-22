"""Git commit search via the public GitHub (and optional GitLab) APIs."""

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

GITHUB_COMMITS_URL = "https://api.github.com/search/commits"
GITLAB_SEARCH_URL = "https://gitlab.com/api/v4/search"


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = env("GITHUB_TOKEN") or env("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _commit_source(
    *,
    sha: str,
    message: str,
    url: Optional[str],
    published: Any,
    author: Optional[str],
    repository: Optional[str],
    provider: str,
) -> PublicSource:
    title = (message or "").strip().splitlines()[0] if message else sha
    content = (message or title).strip()
    if repository:
        content = f"Repository: {repository}\n\n{content}"
    return PublicSource(
        id=generate_id(sha or "commit"),
        title=title[:200] or f"Commit {sha[:12]}",
        content=content,
        source_type=SourceType.GIT_COMMIT,
        url=optional_http_url(url),
        published_date=parse_datetime(published),
        author=author,
        organization=(repository.split("/")[0] if repository and "/" in repository else repository),
        metadata={
            "sha": sha,
            "repository": repository,
            "provider": provider,
        },
        tags=["git", "commit", provider],
    )


def _parse_github_items(data: Any, max_results: int) -> List[PublicSource]:
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    results: List[PublicSource] = []
    for item in items[:max_results]:
        if not isinstance(item, dict):
            continue
        commit = item.get("commit") if isinstance(item.get("commit"), dict) else {}
        author_block = commit.get("author") if isinstance(commit.get("author"), dict) else {}
        repo = item.get("repository") if isinstance(item.get("repository"), dict) else {}
        sha = str(item.get("sha") or "")
        results.append(
            _commit_source(
                sha=sha,
                message=str(commit.get("message") or ""),
                url=item.get("html_url"),
                published=author_block.get("date"),
                author=author_block.get("name"),
                repository=repo.get("full_name"),
                provider="github",
            )
        )
    return results


def _parse_gitlab_items(data: Any, max_results: int) -> List[PublicSource]:
    if not isinstance(data, list):
        return []
    results: List[PublicSource] = []
    for item in data[:max_results]:
        if not isinstance(item, dict):
            continue
        sha = str(item.get("id") or item.get("short_id") or "")
        project = item.get("project_id")
        web_url = item.get("web_url")
        results.append(
            _commit_source(
                sha=sha,
                message=str(item.get("message") or item.get("title") or ""),
                url=web_url,
                published=item.get("created_at") or item.get("committed_date"),
                author=item.get("author_name"),
                repository=str(project) if project is not None else None,
                provider="gitlab",
            )
        )
    return results


async def search_git_commits(
    query: str,
    max_results: int = 100,
    platforms: List[str] | None = None,
) -> List[PublicSource]:
    """Search public git commits.

    GitHub commit search is the default (optional ``GITHUB_TOKEN`` raises the
    rate limit). GitLab global search only runs when ``GITLAB_TOKEN`` is set —
    unauthenticated ``/search`` is not a public endpoint.
    """
    query = (query or "").strip()
    if not query:
        return []
    max_results = max(1, min(int(max_results), 100))
    wanted = {(p or "").strip().lower() for p in (platforms or ["github", "gitlab"])}
    results: List[PublicSource] = []

    if not wanted or "github" in wanted:
        data = await get_json(
            GITHUB_COMMITS_URL,
            params={"q": query, "per_page": str(max_results), "sort": "committer-date"},
            headers=_github_headers(),
        )
        results.extend(_parse_github_items(data, max_results))

    gitlab_token = env("GITLAB_TOKEN")
    if "gitlab" in wanted and gitlab_token:
        data = await get_json(
            GITLAB_SEARCH_URL,
            params={"scope": "commits", "search": query, "per_page": str(max_results)},
            headers={"PRIVATE-TOKEN": gitlab_token},
        )
        results.extend(_parse_gitlab_items(data, max_results))
    elif "gitlab" in wanted and not gitlab_token:
        logger.info("Skipping GitLab commit search (set GITLAB_TOKEN to enable)")

    return results[:max_results]
