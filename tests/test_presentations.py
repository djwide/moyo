"""Map presentation sidecar keeps canonical claims English."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.presentations import translate_presentations, write_map_presentations


def test_english_skips_translation(tmp_path: Path) -> None:
    report = {"findings_all": [{"claim_id": "C0001", "claim": "Acme opened an office."}]}
    (tmp_path / "report_data.json").write_text(json.dumps(report), encoding="utf-8")
    assert write_map_presentations(tmp_path, "English", complete=lambda _prompt: "{}") is None
    assert not (tmp_path / "presentations.json").exists()


def test_sidecar_keys_claims_and_sections(tmp_path: Path) -> None:
    report = {
        "findings_all": [
            {"claim_id": "C0001", "claim": "Acme opened an office.", "summary": "Berlin office"},
        ]
    }
    layout = {"nodes": [{"id": "section:0", "kind": "section", "title": "Corporate filings"}]}
    (tmp_path / "report_data.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "tree_layout.json").write_text(json.dumps(layout), encoding="utf-8")

    def complete(prompt: str) -> str:
        assert "French" in prompt
        return json.dumps(
            {
                "C0001": {"claim": "Acme a ouvert un bureau.", "title": None, "summary": "Bureau de Berlin"},
                "section:corporate filings": {
                    "claim": "Dépôts d'entreprise",
                    "title": "Dépôts d'entreprise",
                    "summary": None,
                },
            }
        )

    dest = write_map_presentations(tmp_path, "French", complete=complete)
    assert dest is not None
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["claims"]["C0001"]["claim"] == "Acme a ouvert un bureau."
    assert payload["sections"]["corporate filings"]["title"] == "Dépôts d'entreprise"
    assert translate_presentations("English", {"C0001": {"claim": "x"}}, complete=lambda _p: "{}") == {}
