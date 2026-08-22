from __future__ import annotations

import json
from pathlib import Path

from moyo.publicside.gatherpublicsources.extract import (
    EXTRACTED_FILE_NAME,
    extract_from_sources,
    extract_relevant_passages,
    load_extracted_sources,
    parse_passage_payload,
    save_extracted,
)
from moyo.publicside.gatherpublicsources.schema import PublicSource, SourceType


def test_parse_passage_payload():
    raw = """```json
    {"phrases": [
      {"text": "Acme vault path prod/db/root", "label": "identifier", "reason": "path"},
      {"text": "", "label": "other"},
      {"text": "CEO Jane Doe", "label": "mystery"}
    ]}
    ```"""
    rows = parse_passage_payload(raw)
    assert len(rows) == 2
    assert rows[0]["label"] == "identifier"
    assert rows[1]["label"] == "other"


def _stub_complete(prompt, system=None):
    return json.dumps(
        {
            "phrases": [
                {
                    "text": "Acme vault path prod/db/root",
                    "label": "identifier",
                    "reason": "credential location",
                },
                {
                    "text": "CEO Jane Doe joined in 2019",
                    "label": "personnel",
                    "reason": "named officer",
                },
            ]
        }
    )


def test_extract_appends_labelled_direction():
    seen = {}

    def capture(prompt, system=None):
        seen["prompt"] = prompt
        return json.dumps({"phrases": []})

    extract_relevant_passages(
        "Acme filed patent US123.",
        direction="Focus on credential formats only.",
        complete=capture,
    )
    assert "Acme filed patent US123." in seen["prompt"]
    assert "\ndirection:\nFocus on credential formats only." in seen["prompt"]
    assert seen["prompt"].index("Acme filed patent US123.") < seen["prompt"].index(
        "direction:"
    )


def test_extract_omits_empty_direction():
    seen = {}

    def capture(prompt, system=None):
        seen["prompt"] = prompt
        return json.dumps({"phrases": []})

    extract_relevant_passages("Acme filed patent US123.", direction="  ", complete=capture)
    assert "direction:" not in seen["prompt"]


def test_extract_emits_determinate_window_progress():
    events = []

    def prog(current, total, message=""):
        events.append((current, total, message))

    src = PublicSource(
        id="src_1",
        title="Note",
        content="Acme vault path prod/db/root. CEO Jane Doe joined in 2019.",
        source_type=SourceType.WEB_SEARCH,
    )
    extract_from_sources([src, src], complete=_stub_complete, progress=prog)
    assert events
    totals = {total for _, total, _ in events}
    assert len(totals) == 1
    total = events[0][1]
    assert total >= 1
    assert events[0][0] == 0
    assert events[-1][0] == total
    currents = [c for c, _, _ in events]
    assert currents == sorted(currents)


def test_format_extract_progress_shows_fraction():
    from moyo.publicside.gatherpublicsources.extract import format_extract_progress

    text = format_extract_progress(3, 12, "Source 1/4: Note")
    assert "3/12" in text
    assert "25.0%" in text
    assert "Source 1/4" in text
    raw = PublicSource(
        id="src_1",
        title="Press note",
        content=(
            "Cookie policy. Click here to subscribe.\n\n"
            "Acme vault path prod/db/root was listed in a GitHub gist.\n"
            "CEO Jane Doe joined in 2019.\n"
        ),
        source_type=SourceType.PRESS_RELEASE,
        tags=["press"],
    )
    extracted = extract_from_sources([raw], direction="Keep vault paths.", complete=_stub_complete)
    assert len(extracted) == 2
    texts = [s.content for s in extracted]
    assert any("prod/db/root" in t for t in texts)
    assert any("Jane Doe" in t for t in texts)
    assert all(s.metadata.get("extracted") for s in extracted)
    assert all(s.metadata.get("source_id") == "src_1" for s in extracted)
    assert not any("Cookie policy" in s.content for s in extracted)


def test_save_extracted_writes_direction(tmp_path: Path):
    sources = extract_from_sources(
        [
            PublicSource(
                id="src_1",
                title="Note",
                content="Acme vault path prod/db/root",
                source_type=SourceType.WEB_SEARCH,
            )
        ],
        direction="Keep vault paths.",
        complete=_stub_complete,
    )
    path = save_extracted(tmp_path, sources, direction="Keep vault paths.")
    assert path.name == EXTRACTED_FILE_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["direction"] == "Keep vault paths."
    assert payload["count"] == 2
    assert payload["sources"][0]["content"]
    loaded = load_extracted_sources(tmp_path)
    assert len(loaded) == 2
    assert any("prod/db/root" in s.content for s in loaded)


def test_load_extracted_missing_raises(tmp_path: Path):
    import pytest
    from moyo.publicside.gatherpublicsources.extract import load_extracted_sources

    with pytest.raises(FileNotFoundError, match="Extracted corpus not found"):
        load_extracted_sources(tmp_path)
