import asyncio
import httpx
import pytest

from moyo.publicside.gatherpublicsources.sources.patents import search_patents
from moyo.publicside.gatherpublicsources.sources.press_releases import search_press_releases
from moyo.publicside.gatherpublicsources.sources.git_commits import search_git_commits
from moyo.publicside.gatherpublicsources.sources.conferences import search_conference_talks
from moyo.publicside.gatherpublicsources.sources.leaks import search_leaked_code


@pytest.mark.parametrize(
    "func,response,expected",
    [
        (
            search_patents,
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "p1",
                            "title": "Patent",
                            "abstract": "A",
                            "url": "http://example.com/p1",
                            "date": "2021-01-01",
                        }
                    ]
                },
                request=httpx.Request("GET", "https://example.com/patents"),
            ),
            "Patent",
        ),
        (
            search_git_commits,
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "c1",
                            "message": "Fix bug",
                            "url": "http://example.com/c1",
                            "date": "2021-01-02",
                            "author": "Alice",
                        }
                    ]
                },
                request=httpx.Request("GET", "https://example.com/commits"),
            ),
            "Fix bug",
        ),
        (
            search_conference_talks,
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "t1",
                            "title": "Talk",
                            "abstract": "About stuff",
                            "url": "http://example.com/t1",
                            "date": "2021-01-03",
                        }
                    ]
                },
                request=httpx.Request("GET", "https://example.com/talks"),
            ),
            "Talk",
        ),
    ],
)
def test_json_parsing(monkeypatch, func, response, expected):
    async def mock_get(self, url, params=None):
        return response

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    results = asyncio.run(func("topic"))
    assert len(results) == 1
    assert results[0].title == expected


@pytest.mark.parametrize(
    "func,html,expected",
    [
        (
            search_press_releases,
            "<article><h1>Release</h1><p>Body</p><a href=\"http://example.com/pr1\"></a><time>2021-01-04</time></article>",
            "Release",
        ),
        (
            search_leaked_code,
            "<pre data-title=\"Leak\" data-url=\"http://example.com/l1\" data-date=\"2021-01-05\">secret</pre>",
            "Leak",
        ),
    ],
)
def test_html_parsing(monkeypatch, func, html, expected):
    response = httpx.Response(200, text=html, request=httpx.Request("GET", "https://example.com"))

    async def mock_get(self, url, params=None):
        return response

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    results = asyncio.run(func("topic"))
    assert len(results) == 1
    assert results[0].title == expected


@pytest.mark.parametrize(
    "func",
    [
        search_patents,
        search_press_releases,
        search_git_commits,
        search_conference_talks,
        search_leaked_code,
    ],
)
def test_error_handling(monkeypatch, func):
    async def mock_get(self, url, params=None):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    results = asyncio.run(func("topic"))
    assert results == []

