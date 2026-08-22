from __future__ import annotations

import json
from pathlib import Path

import pytest

from moyo.project import create_project
from moyo.publicside.gatherpublicsources.load import load_public_sources
from moyo.publicside.gatherpublicsources.schema import SourceType
from moyo.publicside.barrierprobe.schema import IndexConfig
from moyo.privateside.mapcorpus.schema import CorpusConfig
from shared_utils.model_config import DEFAULT_MODEL_KEY, DEFAULT_MODEL_NAME


def test_default_embedding_is_bge_base():
    assert DEFAULT_MODEL_KEY == "bge-base"
    assert DEFAULT_MODEL_NAME == "BAAI/bge-base-en-v1.5"
    assert CorpusConfig().embedding_model == DEFAULT_MODEL_NAME
    assert IndexConfig().embedding_model == DEFAULT_MODEL_NAME


def test_load_sources_json_from_gather_crawl(tmp_path: Path):
    topic = tmp_path / "neural_networks"
    topic.mkdir()
    payload = [
        {
            "id": "src_1",
            "title": "A patent",
            "content": "Claim 1 describes a vault path.",
            "source_type": "patent",
        }
    ]
    (topic / "sources.json").write_text(json.dumps(payload), encoding="utf-8")
    (topic / "summary.json").write_text("{}", encoding="utf-8")

    sources = load_public_sources(tmp_path)
    assert len(sources) == 1
    assert sources[0].title == "A patent"
    assert sources[0].source_type == SourceType.PATENT
    assert "vault path" in sources[0].content


def test_load_exploration_md_from_gather_explore(tmp_path: Path):
    slug = tmp_path / "who_killed_jfk"
    slug.mkdir()
    (slug / "exploration.md").write_text(
        "# Who killed JFK?\n\nPublic reporting names the Warren Commission.\n",
        encoding="utf-8",
    )

    sources = load_public_sources(tmp_path)
    assert len(sources) == 1
    assert sources[0].title == "who_killed_jfk"
    assert sources[0].source_type == SourceType.WEB_SEARCH
    assert "Warren Commission" in sources[0].content
    assert sources[0].metadata["kind"] == "exploration.md"


def test_load_both_gather_file_types(tmp_path: Path):
    crawl = tmp_path / "topic"
    crawl.mkdir()
    (crawl / "sources.json").write_text(
        json.dumps(
            [
                {
                    "id": "src_1",
                    "title": "Press note",
                    "content": "Company filed a patent.",
                    "source_type": "press_release",
                }
            ]
        ),
        encoding="utf-8",
    )
    explore = tmp_path / "naive_prompt"
    explore.mkdir()
    (explore / "exploration.md").write_text(
        "LLM findings about the company.\n", encoding="utf-8"
    )

    sources = load_public_sources(tmp_path)
    kinds = {s.source_type for s in sources}
    titles = {s.title for s in sources}
    assert SourceType.PRESS_RELEASE in kinds
    assert SourceType.WEB_SEARCH in kinds
    assert "Press note" in titles
    assert "naive_prompt" in titles


def test_load_skips_extracted_json(tmp_path: Path):
    (tmp_path / "sources.json").write_text(
        json.dumps(
            [
                {
                    "id": "src_1",
                    "title": "A patent",
                    "content": "Claim 1 describes a vault path.",
                    "source_type": "patent",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "extracted.json").write_text(
        json.dumps(
            {
                "count": 1,
                "sources": [
                    {
                        "id": "extracted_1",
                        "title": "Should not load",
                        "content": "duplicate extracted row",
                        "source_type": "web_search",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sources = load_public_sources(tmp_path)
    assert len(sources) == 1
    assert sources[0].title == "A patent"


def test_load_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_public_sources(tmp_path / "missing")


def test_find_sources_dirs_includes_exploration_only(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv("MOYO_PROJECTS_DIR", str(root))
    monkeypatch.delenv("MOYO_PROJECT", raising=False)
    proj = create_project("acme")
    (proj.public_sources_dir / "run1").mkdir()
    (proj.public_sources_dir / "run1" / "exploration.md").write_text(
        "# explore\n", encoding="utf-8"
    )
    dirs = proj.find_sources_dirs()
    assert proj.public_sources_dir in dirs
    assert (proj.public_sources_dir / "run1") in dirs
