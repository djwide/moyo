from moyo.exposure_preview import estimate_exposure_preview


def test_preview_is_stable_and_avoids_word_counts():
    topic = "What's the secret recipe for Brightwell Cola?"
    first = estimate_exposure_preview(topic)
    second = estimate_exposure_preview(f"  {topic}  ")
    assert first["candidatePaths"] == second["candidatePaths"]
    assert first["relationships"] == second["relationships"]
    assert "wordCount" not in first
    assert first["modelsPlus"] >= 8
    assert first["languagesPlus"] >= 4
    assert first["promptCyclesPlus"] >= 3
    assert 22 <= first["candidatePaths"] <= 86
    assert first["estimatedFindings"] == "15–25"
    assert "trade secret" in " ".join(first["sensitiveCategories"])
    assert first["headline"].startswith("We found ")


def test_richer_topics_open_more_paths_than_bare_names():
    rich = estimate_exposure_preview("What manufacturing or clinical capability do Riverbend Therapeutics have?")
    thin = estimate_exposure_preview("Shafiu abubakar")
    assert rich["candidatePaths"] > thin["candidatePaths"]
    assert rich["denseAreas"] >= thin["denseAreas"]
