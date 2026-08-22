from __future__ import annotations

import json
from pathlib import Path

import pytest

from moyo.project import (
    MoyoProject,
    create_project,
    find_phrase_dirs,
    get_project,
    list_projects,
    resolve_phrases_dir,
    resolve_private_index_dir,
    slugify_project_name,
)


@pytest.fixture
def projects_dir(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv("MOYO_PROJECTS_DIR", str(root))
    monkeypatch.delenv("MOYO_PROJECT", raising=False)
    return root


def test_slugify_project_name():
    assert slugify_project_name("Acme Corp") == "acme_corp"
    assert slugify_project_name("  ") == ""


def test_create_project_layout(projects_dir):
    proj = create_project("Acme Corp")
    assert proj.name == "acme_corp"
    assert proj.root == projects_dir / "acme_corp"
    assert proj.phrases_dir.is_dir()
    assert proj.private_index_dir.is_dir()
    assert proj.public_index_dir.is_dir()
    assert proj.public_sources_dir.is_dir()
    assert proj.compare_dir.is_dir()
    meta = json.loads((proj.root / "moyo-project.json").read_text(encoding="utf-8"))
    assert meta["name"] == "acme_corp"


def test_two_projects_have_separate_phrases_and_indexes(projects_dir):
    a = create_project("alpha")
    b = create_project("beta")
    (a.phrases_dir / "corpus.jsonl").write_text(
        '{"text":"alpha secret","status":"approved"}\n', encoding="utf-8"
    )
    (b.phrases_dir / "corpus.jsonl").write_text(
        '{"text":"beta secret","status":"approved"}\n', encoding="utf-8"
    )
    (a.private_index_dir / "alpha.faiss").write_bytes(b"a")
    (b.private_index_dir / "beta.faiss").write_bytes(b"b")

    names = {p.name for p in list_projects()}
    assert names == {"alpha", "beta"}
    assert a.find_phrase_corpus() != b.find_phrase_corpus()
    assert "alpha secret" in a.find_phrase_corpus().read_text(encoding="utf-8")
    assert "beta secret" in b.find_phrase_corpus().read_text(encoding="utf-8")
    assert a.latest_private_index() != b.latest_private_index()


def test_search_finds_nested_corpus_and_faiss(projects_dir):
    proj = create_project("nested")
    nested_phrases = proj.root / "archive" / "old_phrases"
    nested_phrases.mkdir(parents=True)
    (nested_phrases / "pending.jsonl").write_text("{}\n", encoding="utf-8")
    faiss_dir = proj.root / "indexes" / "private" / "nested"
    faiss_dir.mkdir(parents=True)
    (faiss_dir / "nested.faiss").write_bytes(b"idx")
    (proj.public_sources_dir / "run1").mkdir()
    (proj.public_sources_dir / "run1" / "exploration.md").write_text("# hi\n", encoding="utf-8")
    (proj.public_sources_dir / "run1" / "sources.json").write_text("[]\n", encoding="utf-8")

    found = MoyoProject.from_path(proj.root)
    dirs = find_phrase_dirs(found.root)
    assert nested_phrases in dirs or found.phrases_dir in dirs
    assert any(p.name == "nested" for p in found.find_private_indexes())
    assert found.find_explorations()
    assert found.find_sources_dirs()


def test_resolve_requires_project(projects_dir):
    with pytest.raises(ValueError, match="No project selected"):
        resolve_phrases_dir()
    with pytest.raises(ValueError, match="No project selected"):
        resolve_private_index_dir()


def test_resolve_from_project_name(projects_dir):
    create_project("gamma")
    phrases = resolve_phrases_dir(project="gamma")
    private = resolve_private_index_dir(project="gamma")
    assert phrases == projects_dir / "gamma" / "phrases"
    assert private == projects_dir / "gamma" / "indexes" / "private"


def test_get_project_missing(projects_dir):
    with pytest.raises(FileNotFoundError):
        get_project("missing")
    created = get_project("later", create=True)
    assert created.root.exists()
