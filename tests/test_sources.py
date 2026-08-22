import asyncio

import pytest

from moyo.publicside.gatherpublicsources.schema import SourceType
from moyo.publicside.gatherpublicsources.sources.conferences import search_conference_talks
from moyo.publicside.gatherpublicsources.sources.git_commits import search_git_commits
from moyo.publicside.gatherpublicsources.sources.http import parse_datetime, reconstruct_inverted_abstract
from moyo.publicside.gatherpublicsources.sources.leaks import search_leaked_code
from moyo.publicside.gatherpublicsources.sources.patents import search_patents
from moyo.publicside.gatherpublicsources.sources.press_releases import search_press_releases


ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2101.00001</id>
    <title>Talk</title>
    <summary>About stuff</summary>
    <published>2021-01-03T00:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
    <link rel="alternate" href="https://arxiv.org/abs/2101.00001"/>
  </entry>
</feed>
"""


def test_parse_datetime_formats():
    assert parse_datetime("2021-01-04").year == 2021
    assert parse_datetime("20240101T120000Z").day == 1
    assert parse_datetime(None) is None


def test_openalex_abstract_rebuild():
    text = reconstruct_inverted_abstract({"Hello": [0], "world": [1]})
    assert text == "Hello world"


def test_patent_google_parsing(monkeypatch):
    monkeypatch.delenv("PATENTSVIEW_API_KEY", raising=False)
    async def fake_json(url, **kwargs):
        assert "patents.google.com" in url
        return {
            "results": {
                "cluster": [
                    [
                        {
                            "patent": {
                                "publication_number": "US123",
                                "title": "Patent",
                                "snippet": "A",
                                "publication_date": "2021-01-01",
                            }
                        }
                    ]
                ]
            }
        }

    monkeypatch.setattr(
        "moyo.publicside.gatherpublicsources.sources.patents.get_json",
        fake_json,
    )
    results = asyncio.run(search_patents("topic"))
    assert len(results) == 1
    assert results[0].title == "Patent"
    assert results[0].source_type == SourceType.PATENT
    assert "US123" in str(results[0].url)


def test_press_gdelt_parsing(monkeypatch):
    async def fake_json(url, **kwargs):
        assert "gdeltproject.org" in url
        return {
            "articles": [
                {
                    "url": "https://www.prnewswire.com/news-releases/example",
                    "title": "Release",
                    "seendate": "20210104T000000Z",
                    "domain": "prnewswire.com",
                }
            ]
        }

    monkeypatch.setattr(
        "moyo.publicside.gatherpublicsources.sources.press_releases.get_json",
        fake_json,
    )
    results = asyncio.run(search_press_releases("topic"))
    assert len(results) == 1
    assert results[0].title == "Release"
    assert results[0].source_type == SourceType.PRESS_RELEASE


def test_git_github_parsing(monkeypatch):
    async def fake_json(url, **kwargs):
        assert "api.github.com/search/commits" in url
        return {
            "items": [
                {
                    "sha": "abc123",
                    "html_url": "https://github.com/org/repo/commit/abc123",
                    "commit": {
                        "message": "Fix bug",
                        "author": {"name": "Alice", "date": "2021-01-02T00:00:00Z"},
                    },
                    "repository": {"full_name": "org/repo"},
                }
            ]
        }

    monkeypatch.setattr(
        "moyo.publicside.gatherpublicsources.sources.git_commits.get_json",
        fake_json,
    )
    results = asyncio.run(search_git_commits("topic", platforms=["github"]))
    assert len(results) == 1
    assert results[0].title == "Fix bug"
    assert results[0].author == "Alice"


def test_conference_arxiv_parsing(monkeypatch):
    async def fake_text(url, **kwargs):
        assert "arxiv.org" in url
        return ARXIV_ATOM

    async def fake_json(url, **kwargs):
        return {"results": []}

    monkeypatch.setattr(
        "moyo.publicside.gatherpublicsources.sources.conferences.get_text",
        fake_text,
    )
    monkeypatch.setattr(
        "moyo.publicside.gatherpublicsources.sources.conferences.get_json",
        fake_json,
    )
    results = asyncio.run(search_conference_talks("topic", sources=["arxiv"]))
    assert len(results) == 1
    assert results[0].title == "Talk"


def test_leaks_nvd_parsing(monkeypatch):
    async def fake_json(url, **kwargs):
        if "nvd.nist.gov" in url:
            return {
                "vulnerabilities": [
                    {
                        "cve": {
                            "id": "CVE-2021-0001",
                            "published": "2021-01-05T00:00:00.000",
                            "descriptions": [
                                {"lang": "en", "value": "Leak"}
                            ],
                            "references": [
                                {"url": "https://nvd.nist.gov/vuln/detail/CVE-2021-0001"}
                            ],
                        }
                    }
                ]
            }
        return []

    monkeypatch.setattr(
        "moyo.publicside.gatherpublicsources.sources.leaks.get_json",
        fake_json,
    )
    results = asyncio.run(search_leaked_code("topic", sources=["nvd"]))
    assert len(results) == 1
    assert results[0].title == "CVE-2021-0001"
    assert results[0].source_type == SourceType.LEAKED_CODE


@pytest.mark.parametrize(
    "func,module",
    [
        (search_patents, "moyo.publicside.gatherpublicsources.sources.patents"),
        (search_press_releases, "moyo.publicside.gatherpublicsources.sources.press_releases"),
        (search_git_commits, "moyo.publicside.gatherpublicsources.sources.git_commits"),
        (search_leaked_code, "moyo.publicside.gatherpublicsources.sources.leaks"),
    ],
)
def test_json_error_handling(monkeypatch, func, module):
    async def boom(*args, **kwargs):
        return None

    monkeypatch.setattr(f"{module}.get_json", boom)
    kwargs = {}
    if func is search_git_commits:
        kwargs["platforms"] = ["github"]
    if func is search_leaked_code:
        kwargs["sources"] = ["nvd"]
    results = asyncio.run(func("topic", **kwargs))
    assert results == []


def test_arxiv_error_handling(monkeypatch):
    async def boom(*args, **kwargs):
        return ""

    monkeypatch.setattr(
        "moyo.publicside.gatherpublicsources.sources.conferences.get_text",
        boom,
    )
    results = asyncio.run(search_conference_talks("topic", sources=["arxiv"]))
    assert results == []
