from core.scenario_support import classroom_scenario_support


def _all_supported(species, population, preset):
    del species, population
    return {
        "spring migration": {
            f"outcome_for_{preset.key}": {
                "kind": "continuous",
                "baseline": 1.0,
                "scenario": 2.0,
                "delta": 1.0,
            }
        }
    }


def test_support_matrix_covers_every_promised_non_baseline_scenario():
    reports = classroom_scenario_support(_all_supported)

    assert len(reports) == 6
    assert all(report.supported for report in reports)
    assert {report.scenario_key for report in reports} == {
        "helpful-tailwind",
        "strong-headwind",
        "warmer-spring",
        "less-nearby-sea-ice",
        "stronger-surface-winds",
        "calmer-surface-winds",
    }


def test_empty_effects_fail_the_support_contract():
    reports = classroom_scenario_support(
        lambda species, population, preset: (
            {} if preset.key == "strong-headwind" else _all_supported(species, population, preset)
        )
    )

    unsupported = [report for report in reports if not report.supported]
    assert [report.scenario_key for report in unsupported] == ["strong-headwind"]
