from __future__ import annotations

import json
from pathlib import Path

import pytest

from moyo.compare.naive import (
    CHAR_BUDGET,
    PackedSide,
    assemble_result,
    load_result,
    pack_compare_prompt,
    parse_compare_payload,
    private_only_by_label,
    resolve_public_pack,
    run_naive_compare,
    save_result,
)
from moyo.privateside.phrases.schema import PhraseRecord, phrase_id
from moyo.privateside.phrases.store import PhraseStore
from moyo.project import MoyoProject


def _phrase(text: str, label: str = "other") -> PhraseRecord:
    return PhraseRecord(
        id=phrase_id(text),
        text=text,
        label=label,
        status="approved",
        source="manual",
    )


def test_parse_compare_json_payload():
    raw = """```json
    {
      "headline": "Vault paths stay private; founding year is public.",
      "only_private": [
        {"id": "ph_aaaaaaaaaaaa", "text": "/secret/app/db", "reason": "not in pack"}
      ],
      "overlap": [
        {"id": "ph_bbbbbbbbbbbb", "text": "founded 2019", "quote": "founded in 2019"}
      ],
      "only_public": [{"text": "rumored Series B", "quote": "raised a Series B"}],
      "caveats": ["public pack truncated"]
    }
    ```"""
    parsed = parse_compare_payload(raw)
    assert parsed is not None
    assert "Vault paths" in parsed["headline"]
    assert parsed["only_private"][0]["text"] == "/secret/app/db"
    assert parsed["overlap"][0]["quote"] == "founded in 2019"
    assert parsed["only_public"][0]["text"] == "rumored Series B"
    assert parsed["caveats"] == ["public pack truncated"]


def test_parse_rejects_empty_payload():
    assert parse_compare_payload("not json") is None
    assert parse_compare_payload("{}") is None


def test_assemble_matches_ids_and_fills_unscored():
    vault = _phrase("prod vault path /secret/app/db", "credential")
    year = _phrase("founded 2019, Austin", "other")
    extra = _phrase("codename HALCYON", "project")
    parsed = {
        "headline": "Most identifiers stay private.",
        "only_private": [{"id": vault.id, "text": vault.text, "quote": "", "reason": "absent"}],
        "overlap": [{"id": year.id, "text": year.text, "quote": "Austin-based, 2019", "reason": ""}],
        "only_public": [{"text": "rumored Series B", "quote": "", "reason": ""}],
        "caveats": [],
    }
    dummy = PackedSide(kind="exploration", path="x.md", text="x", chars=1, truncated=False)
    result = assemble_result(
        [vault, year, extra],
        parsed,
        packed_private=dummy,
        packed_public=dummy,
        raw="{}",
    )
    by_id = {row.id: row for row in result.phrase_rows}
    assert by_id[vault.id].verdict == "private-only"
    assert by_id[vault.id].label == "credential"
    assert by_id[year.id].verdict == "overlap"
    assert by_id[year.id].quote == "Austin-based, 2019"
    assert by_id[extra.id].verdict == "unscored"
    counts = private_only_by_label(result)
    assert counts["credential"] == 1
    assert counts["project"] == 0


def test_pack_prefers_extracted_over_exploration(tmp_path: Path):
    public = tmp_path / "public_sources" / "run1"
    public.mkdir(parents=True)
    (public / "exploration.md").write_text("# Raw explore\nFounded in 2019.\n", encoding="utf-8")
    (tmp_path / "public_sources" / "extracted.json").write_text(
        json.dumps(
            {
                "direction": "Keep vault paths.",
                "count": 1,
                "sources": [
                    {
                        "id": "ex_1",
                        "title": "Extracted",
                        "content": "Acme vault path prod/db/root",
                        "source_type": "web_search",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    packed = resolve_public_pack(MoyoProject.from_path(tmp_path))
    assert packed.kind == "extracted"
    assert "prod/db/root" in packed.text
    assert "Founded in 2019" not in packed.text


def test_pack_prefers_exploration_over_sources(tmp_path: Path):
    public = tmp_path / "public_sources" / "run1"
    public.mkdir(parents=True)
    (public / "exploration.md").write_text("# Public notes\nFounded in 2019.\n", encoding="utf-8")
    (public / "sources.json").write_text(
        json.dumps([{"title": "Ignored", "content": "should not be used"}]),
        encoding="utf-8",
    )
    project = MoyoProject.from_path(tmp_path)
    packed = resolve_public_pack(project)
    assert packed.kind == "exploration"
    assert "Founded in 2019" in packed.text
    assert "should not be used" not in packed.text


def test_pack_falls_back_to_sources_json(tmp_path: Path):
    public = tmp_path / "public_sources" / "crawl"
    public.mkdir(parents=True)
    (public / "sources.json").write_text(
        json.dumps(
            [
                {
                    "title": "Press note",
                    "url": "https://example.com/n",
                    "content": "Company expanded to Europe.",
                }
            ]
        ),
        encoding="utf-8",
    )
    packed = resolve_public_pack(MoyoProject.from_path(tmp_path))
    assert packed.kind == "sources"
    assert "expanded to Europe" in packed.text


def test_pack_truncates_public_side():
    phrases = [_phrase("secret token abc", "credential")]
    public = PackedSide(
        kind="exploration",
        path="big.md",
        text="PUBLIC " * 5000,
        chars=0,
        truncated=False,
        item_count=1,
    )
    packed_priv, packed_pub, prompt = pack_compare_prompt(
        phrases, public, char_budget=4_000
    )
    assert packed_pub.truncated
    assert packed_pub.omitted_chars > 0
    assert "truncated" in packed_pub.text
    assert packed_priv.item_count == 1
    assert len(prompt) <= 4_000 + 200


def test_run_naive_compare_with_stub(tmp_path: Path):
    project = MoyoProject.from_path(tmp_path).ensure()
    store = PhraseStore(project.phrases_dir)
    vault = store.add_manual("prod vault path /secret/app/db", "credential")
    year = store.add_manual("founded 2019, Austin", "other")
    assert vault is not None and year is not None
    expl = project.public_sources_dir / "topic" / "exploration.md"
    expl.parent.mkdir(parents=True)
    expl.write_text(
        "The company was founded in 2019 in Austin and expanded publicly.\n",
        encoding="utf-8",
    )

    def complete(prompt, system=None):
        assert "prod vault path" in prompt
        assert "founded in 2019" in prompt.lower() or "PUBLIC PACK" in prompt
        return json.dumps(
            {
                "headline": "The vault path is still private; founding year is already public.",
                "only_private": [
                    {
                        "id": vault.id,
                        "text": vault.text,
                        "reason": "no vault path in public pack",
                    }
                ],
                "overlap": [
                    {
                        "id": year.id,
                        "text": year.text,
                        "quote": "founded in 2019 in Austin",
                    }
                ],
                "only_public": [{"text": "expanded publicly", "quote": "expanded publicly"}],
                "caveats": [],
            }
        )

    result = run_naive_compare(project=project, complete=complete)
    assert "vault path is still private" in result.headline
    assert any(r.verdict == "private-only" and r.label == "credential" for r in result.phrase_rows)
    assert any(r.verdict == "overlap" for r in result.phrase_rows)
    saved = load_result(project)
    assert saved is not None
    assert saved.headline == result.headline
    assert (project.compare_dir / "last.json").is_file()


def test_run_naive_compare_requires_public_pack(tmp_path: Path):
    project = MoyoProject.from_path(tmp_path).ensure()
    PhraseStore(project.phrases_dir).add_manual("codename HALCYON", "project")
    with pytest.raises(ValueError, match="extracted.json"):
        run_naive_compare(
            project=project,
            sources_dir=project.public_sources_dir,
            complete=lambda *a, **k: "{}",
        )


def test_run_naive_compare_requires_approved_phrases(tmp_path: Path):
    project = MoyoProject.from_path(tmp_path).ensure()
    (project.public_sources_dir / "exploration.md").write_text("public\n", encoding="utf-8")
    with pytest.raises(ValueError, match="approved"):
        run_naive_compare(project=project, complete=lambda *a, **k: "{}")


def test_save_and_load_roundtrip(tmp_path: Path):
    project = MoyoProject.from_path(tmp_path).ensure()
    dummy = PackedSide(kind="exploration", path="x.md", text="x", chars=1, truncated=False)
    result = assemble_result(
        [_phrase("alpha secret", "project")],
        {
            "headline": "Alpha stays private.",
            "only_private": [],
            "overlap": [],
            "only_public": [],
            "caveats": ["n=1"],
        },
        packed_private=dummy,
        packed_public=dummy,
        raw="{}",
        char_budget=CHAR_BUDGET,
    )
    path = save_result(project, result)
    loaded = load_result(project)
    assert loaded is not None
    assert loaded.headline == "Alpha stays private."
    assert loaded.caveats == ["n=1"]
    assert path == project.compare_dir / "last.json"
