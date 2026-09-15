from moyo.exposure_preview import estimate_exposure_preview


def test_preview_is_stable_and_avoids_word_counts():
    topic = "What's the secret recipe for Brightwell Cola?"
    first = estimate_exposure_preview(topic)
    second = estimate_exposure_preview(f"  {topic}  ")
    assert first["snapshot"] == second["snapshot"]
    assert first["basis"] == second["basis"]
    assert first["redTeam"] == second["redTeam"]
    assert "wordCount" not in first
    assert first["modelsPlus"] >= 8
    assert first["promptCyclesPlus"] >= 3
    assert first["strategiesPlus"] == 3
    assert 3 <= first["snapshot"]["notableExposures"] <= 10
    assert 8 <= first["basis"]["inventoryFindings"] <= 25
    assert first["snapshot"]["notableExposures"] <= first["basis"]["inventoryFindings"]
    assert first["snapshot"]["citedSources"] == first["basis"]["citedSources"]
    assert first["snapshot"]["claims"] == first["basis"]["claims"]
    assert first["snapshot"]["corroborated"] == first["basis"]["corroborated"]
    assert first["basis"]["claims"] > first["basis"]["inventoryFindings"]
    assert first["basis"]["corroborated"] <= first["basis"]["inventoryFindings"]
    assert 3 <= first["basis"]["exposureChains"] <= 5
    assert 4 <= first["redTeam"]["potentialBreaches"] <= 10
    assert first["redTeam"]["highRiskBreaches"] <= first["redTeam"]["potentialBreaches"]
    assert first["redTeam"]["concentratedMatches"] <= first["redTeam"]["potentialBreaches"]
    assert 2 <= first["redTeam"]["clusters"] <= 12
    assert 8 <= first["redTeam"]["candidatePaths"] <= 24
    assert "trade secret" in " ".join(first["sensitiveCategories"])
    assert "Snapshot" in first["headline"]
    assert "Basis Report" in first["headline"]


def test_richer_topics_estimate_a_larger_basis_inventory():
    rich = estimate_exposure_preview(
        "What manufacturing or clinical capability do Riverbend Therapeutics have?"
    )
    thin = estimate_exposure_preview("Shafiu abubakar")
    assert rich["basis"]["inventoryFindings"] >= thin["basis"]["inventoryFindings"]
    assert rich["basis"]["citedSources"] >= thin["basis"]["citedSources"]
    assert rich["basis"]["claims"] >= thin["basis"]["claims"]
    assert rich["snapshot"]["notableExposures"] >= thin["snapshot"]["notableExposures"]
    assert rich["redTeam"]["potentialBreaches"] >= thin["redTeam"]["potentialBreaches"]
