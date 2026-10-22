"""Map scans collapse once against claims already on the map."""

from __future__ import annotations

from pipeline.map_cluster import cluster_against_map


def _claim(claim_id: str, text: str, model: str) -> dict:
    return {
        "claim_id": claim_id,
        "claim": text,
        "source_model": model,
        "source_models": [model],
        "citations": [f"https://example.com/{claim_id}"],
        "sensitivity": 3,
        "specificity": 3,
        "novelty": 3,
        "interestingness": 3,
        "confidence": 2,
    }


def test_exact_match_updates_the_anchor_and_drops_the_new_claim() -> None:
    claims, _clusters, collapse = cluster_against_map(
        [_claim("C0001", "Acme opened an office in Berlin.", "anthropic:claude")],
        [
            {
                "nodeId": "claim_existing",
                "claim": "Acme opened an office in Berlin.",
                "sourceModels": ["openai:gpt-4o"],
            }
        ],
        group_fn=lambda _rows: [],
    )
    assert claims == []
    assert collapse["duplicate_count"] == 1
    assert collapse["anchor_updates"] == [
        {
            "id": "claim_existing",
            "source_models": ["openai:gpt-4o", "anthropic:claude"],
            "corroboration": 2,
        }
    ]


def test_section_anchor_blocks_a_duplicate_without_taking_models() -> None:
    _claims, _clusters, collapse = cluster_against_map(
        [_claim("C0001", "Banking relationships", "openai:gpt-4o")],
        [{"nodeId": "section_bank", "claim": "Banking relationships", "section": True, "sourceModels": []}],
        group_fn=lambda _rows: [],
    )
    assert collapse["anchor_updates"] == []
    assert collapse["duplicate_count"] == 1


def test_semantic_group_merges_paraphrases_and_keeps_first_seen_models() -> None:
    def group(rows: list[dict]) -> list[list[str]]:
        ids = [row["claim_id"] for row in rows]
        return [ids]

    claims, _clusters, collapse = cluster_against_map(
        [
            _claim("C0001", "Acme opened a Cayman office in 2019.", "openai:gpt-4o"),
            _claim("C0002", "In 2019 Acme opened its Cayman office.", "anthropic:claude"),
        ],
        [],
        group_fn=group,
    )
    assert len(claims) == 1
    assert claims[0]["source_models"] == ["openai:gpt-4o", "anthropic:claude"]
    assert collapse["duplicate_count"] == 1
    assert collapse["collapsed"] is True


def test_word_overlap_stays_split_when_the_model_does_not_group_it() -> None:
    def singletons(rows: list[dict]) -> list[list[str]]:
        return [[row["claim_id"]] for row in rows]

    left = "Acme opened a large regional office in central Berlin during the spring and hired local staff."
    right = left + " last year."
    claims, _clusters, collapse = cluster_against_map(
        [_claim("C0001", left, "openai:gpt-4o"), _claim("C0002", right, "anthropic:claude")],
        [],
        group_fn=singletons,
    )
    assert len(claims) == 2
    assert collapse["duplicate_count"] == 0
