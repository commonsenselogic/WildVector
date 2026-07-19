from core.classroom_scenarios import presets_for_species


def test_each_featured_species_gets_only_relevant_bounded_presets():
    vulture = presets_for_species("Cathartes aura")
    fox = presets_for_species("Vulpes lagopus")
    whale = presets_for_species("Balaenoptera musculus")

    assert [preset.key for preset in vulture] == [
        "typical", "helpful-tailwind", "strong-headwind"
    ]
    assert [preset.key for preset in fox] == [
        "typical", "warmer-spring", "less-nearby-sea-ice"
    ]
    assert [preset.key for preset in whale] == [
        "typical", "stronger-surface-winds", "calmer-surface-winds"
    ]
    assert vulture[1].scenario.tailwind_change_kmh == 12
    assert fox[2].scenario.sea_ice_concentration_change == -0.25
    assert whale[1].scenario.wind_speed_change_kmh == 10


def test_unknown_species_exposes_no_unsupported_simulation():
    presets = presets_for_species("Unknown species")
    assert len(presets) == 1
    assert presets[0].scenario.is_typical


def test_every_student_scenario_has_a_plain_language_cause_and_observation_prompt():
    presets = {
        preset.key: preset
        for species in ("Cathartes aura", "Vulpes lagopus", "Balaenoptera musculus")
        for preset in presets_for_species(species)
    }

    assert set(presets) == {
        "typical",
        "helpful-tailwind",
        "strong-headwind",
        "warmer-spring",
        "less-nearby-sea-ice",
        "stronger-surface-winds",
        "calmer-surface-winds",
    }
    for preset in presets.values():
        assert preset.cause_label
        assert preset.cause_value
        assert preset.observation_prompt.startswith(("Notice", "Predict"))
