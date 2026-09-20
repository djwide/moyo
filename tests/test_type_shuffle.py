"""Type-similar shuffle is optional and domain-agnostic."""

from moyo.publicside.barrierprobe.fuzz_orchestrator import build_call_plan
from moyo.publicside.barrierprobe.llm_fuzzer import (
    LLMFuzzer,
    LLMFuzzerConfig,
    OPTIONAL_FUZZ_STRATEGIES,
    WHITEBOX_FUZZ_STRATEGIES,
    normalize_fuzz_strategies,
    strategies_for_fuzz_mode,
)
from moyo.publicside.barrierprobe.type_shuffle import (
    apply_shuffle,
    collect_shuffle_variants,
    discover_slots,
)


def test_shuffle_is_not_a_default_strategy():
    assert "shuffle" not in strategies_for_fuzz_mode("basic")
    assert "shuffle" not in strategies_for_fuzz_mode("multilingual")
    assert "translate" not in strategies_for_fuzz_mode("basic")
    assert "translate" not in strategies_for_fuzz_mode("multilingual")
    assert "shuffle" not in WHITEBOX_FUZZ_STRATEGIES
    assert "shuffle" in OPTIONAL_FUZZ_STRATEGIES
    assert "shuffle" not in normalize_fuzz_strategies(None, fuzz_mode="basic")
    assert normalize_fuzz_strategies(["shuffle"]) == ["shuffle"]
    assert strategies_for_fuzz_mode("basic") == ["paraphrase", "abstract", "summarize"]


def test_call_plan_omits_shuffle_unless_requested():
    default = build_call_plan(WHITEBOX_FUZZ_STRATEGIES)
    assert all(call.strategy != "shuffle" for call in default)
    plan = build_call_plan(["paraphrase", "shuffle"], calls_per_strategy=2)
    assert [call.label for call in plan] == [
        "paraphrase",
        "paraphrase",
        "shuffle",
        "shuffle",
    ]
    assert [call.repeat_index for call in plan if call.strategy == "shuffle"] == [0, 1]


def test_color_list_permutes_and_keeps_frame():
    phrase = "red, green, and blue"
    variants = collect_shuffle_variants(phrase)
    assert variants
    assert phrase.lower() not in {item.lower() for item in variants}
    for item in variants:
        assert "and" in item.lower()
        for color in ("red", "green", "blue"):
            assert color in item.lower()
    assert any(item.lower().startswith("green") for item in variants)


def test_list_substitutes_from_similar_texts_keeps_an_anchor():
    phrase = "red, green, and blue"
    variants = collect_shuffle_variants(
        phrase,
        similar_texts=["yellow, orange, and purple"],
    )
    joined = " ".join(variants).lower()
    assert any(color in joined for color in ("yellow", "orange", "purple"))
    anchored = [
        item
        for item in variants
        if "yellow" in item.lower() or "orange" in item.lower() or "purple" in item.lower()
    ]
    assert anchored
    for item in anchored:
        originals = sum(color in item.lower() for color in ("red", "green", "blue"))
        assert originals >= 2


def test_function_words_are_not_slots():
    phrase = "the vault path is prod/secrets/db-root"
    _regions, slots = discover_slots(phrase)
    texts = {slot.text.lower() for slot in slots}
    assert "the" not in texts
    assert "is" not in texts
    assert any("/" in slot.text for slot in slots)


def test_path_substitutes_from_similar_texts():
    phrase = "the vault path is prod/secrets/db-root"
    variants = collect_shuffle_variants(
        phrase,
        similar_texts=["staging/secrets/db-root is the staging mount"],
    )
    assert any("staging/secrets/db-root" in item for item in variants)
    assert all(item.lower().startswith("the vault path is") for item in variants)


def test_two_ids_swap():
    phrase = "tenant ids tnt_orion_9c2f and tnt_orion_1abc"
    variants = collect_shuffle_variants(phrase)
    swapped = [
        item
        for item in variants
        if "tnt_orion_9c2f" in item
        and "tnt_orion_1abc" in item
        and item.index("tnt_orion_1abc") < item.index("tnt_orion_9c2f")
    ]
    assert swapped


def test_amounts_and_percents_stay_in_kind():
    phrase = "revenue $47.2 million at 18 percent"
    variants = collect_shuffle_variants(
        phrase,
        similar_texts=["forecast $50 million at 21 percent"],
    )
    assert any("$50 million" in item for item in variants)
    assert any("21 percent" in item for item in variants)
    assert not any(item.startswith("revenue 18 percent") for item in variants)


def test_repeat_index_walks_distinct_variants():
    phrase = "oak, maple, and pine"
    first = apply_shuffle(phrase, repeat_index=0)
    second = apply_shuffle(phrase, repeat_index=1)
    assert first
    assert second
    assert first != second
    third = apply_shuffle(phrase, repeat_index=len(collect_shuffle_variants(phrase)))
    assert third == first


def test_plain_sentence_without_slots_yields_nothing():
    phrase = "this reports the status as generally available"
    assert collect_shuffle_variants(phrase) == []


def test_modify_text_shuffle_does_not_call_llm(monkeypatch):
    fuzzer = LLMFuzzer(LLMFuzzerConfig(llm_provider="test"))

    def boom(*_a, **_k):
        raise AssertionError("LLM should not run for shuffle")

    monkeypatch.setattr(fuzzer, "query_llm", boom)
    monkeypatch.setattr(
        fuzzer, "_embed_texts", lambda texts, model: [[0.0, 1.0]] * len(texts)
    )
    raw, prompt, response = fuzzer.modify_text(
        "red, green, and blue", "shuffle", target_concept="colors"
    )
    assert raw
    assert raw.lower() != "red, green, and blue"
    assert "shuffle" in prompt.lower()
    assert response == raw

    other, _, _ = fuzzer.modify_text(
        "red, green, and blue", "shuffle", repeat_index=1, target_concept="colors"
    )
    assert other
    assert other != raw
