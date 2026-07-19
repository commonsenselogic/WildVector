import numpy as np
import pandas as pd

from core.population import build_population_corridor, simulate_population_scenario
from core.scenario import WeatherScenario, typical_route_direction
from core.visualization import (
    _extend_route_to_known_terminals,
    build_animation_track,
    build_population_deck,
)


def population_frame(years=4, animals=4, points=16):
    rows = []
    for year in range(2018, 2018 + years):
        for animal in range(animals):
            times = pd.date_range(f"{year}-03-01", periods=points, freq="5D", tz="UTC")
            for index, timestamp in enumerate(times):
                progress = index / (points - 1)
                rows.append(
                    {
                        "species": "Testus migratorius",
                        "population": "Test flyway",
                        "movement_type": "aerial",
                        "season": "spring migration",
                        "year": year,
                        "animal_id": f"bird-{animal}",
                        "timestamp_utc": timestamp,
                        "latitude": 20 + 20 * progress + animal * 0.15 + (year - 2018) * 0.05,
                        "longitude": -100 + 12 * progress + np.sin(progress * np.pi) * animal * 0.2,
                    }
                )
    return pd.DataFrame(rows)


def test_population_corridor_combines_animals_and_years():
    frame = population_frame()
    corridor = build_population_corridor(frame, bins=20)
    assert len(corridor.corridor) == 20
    assert corridor.animals == 4
    assert corridor.years == 4
    assert len(corridor.journeys) == 16
    assert corridor.journeys.representative.sum() == 1
    assert corridor.corridor.spread_p90_km.gt(0).all()


def test_population_weather_changes_corridor_and_retains_variation():
    frame = population_frame()
    representative = build_population_corridor(frame).paths
    route = representative[representative.representative].copy()
    route["timestamp_utc"] = pd.date_range("2020-01-01", periods=len(route), freq="h", tz="UTC")
    direction = typical_route_direction(route)
    result = simulate_population_scenario(
        frame,
        WeatherScenario(wind_speed_change_kmh=15, wind_direction_deg=direction),
        "aerial",
        bins=20,
    )
    assert result.changed.animals == result.baseline.animals
    assert result.changed.years == result.baseline.years
    assert len(result.changed.paths) == len(result.baseline.paths)
    baseline_coordinates = result.baseline.paths[["journey_id", "progress", "latitude", "longitude"]]
    changed_coordinates = result.changed.paths[["journey_id", "progress", "latitude", "longitude"]]
    pd.testing.assert_frame_equal(baseline_coordinates, changed_coordinates)
    assert result.changed.journeys.scenario_weight.nunique() > 1

    route_relative = simulate_population_scenario(
        frame, WeatherScenario(wind_speed_change_kmh=12), "aerial", bins=20
    )
    assert route_relative.changed.journeys.scenario_weight.nunique() > 1


def test_spring_and_fall_are_kept_as_separate_journeys():
    spring = population_frame(years=1, animals=1)
    fall = spring.copy()
    fall["season"] = "fall migration"
    fall["timestamp_utc"] = fall.timestamp_utc + pd.Timedelta(days=180)
    fall["latitude"] = fall.latitude.iloc[::-1].to_numpy()
    fall["longitude"] = fall.longitude.iloc[::-1].to_numpy()
    corridor = build_population_corridor(pd.concat([spring, fall], ignore_index=True))
    assert len(corridor.journeys) == 2
    assert set(corridor.journeys.season) == {"spring migration", "fall migration"}


def test_unvalidated_controls_have_zero_effect_and_activated_outcomes_only_reweight():
    frame = population_frame()
    blocked = simulate_population_scenario(
        frame,
        WeatherScenario(wind_speed_change_kmh=20, temperature_change_c=8),
        "aerial",
        activated_effects={},
    )
    assert blocked.changed.journeys.scenario_weight.eq(1).all()
    activated = simulate_population_scenario(
        frame,
        WeatherScenario(temperature_change_c=4),
        "aerial",
        activated_effects={
            "spring migration": {
                "departure_date": {"kind": "continuous", "delta": 2.0, "unit": "days"}
            }
        },
    )
    assert activated.changed.journeys.scenario_weight.nunique() > 1
    pd.testing.assert_frame_equal(
        activated.baseline.paths[["latitude", "longitude"]],
        activated.changed.paths[["latitude", "longitude"]],
    )


def test_activated_corridor_probabilities_reweight_matching_recorded_corridors():
    frame = population_frame(years=3, animals=6)
    frame.loc[frame.animal_id.isin(["bird-3", "bird-4", "bird-5"]), "longitude"] += 8
    result = simulate_population_scenario(
        frame,
        WeatherScenario(wind_speed_change_kmh=10),
        "aerial",
        activated_effects={
            "spring migration": {
                "corridor_choice": {
                    "kind": "classification",
                    "baseline": {"corridor 1": 0.5, "corridor 2": 0.5},
                    "scenario": {"corridor 1": 0.8, "corridor 2": 0.2},
                }
            }
        },
    )
    assert set(result.changed.journeys.corridor_choice) == {"corridor 1", "corridor 2"}
    weights = result.changed.journeys.groupby("corridor_choice").scenario_weight.mean()
    assert weights["corridor 1"] > weights["corridor 2"]
    pd.testing.assert_frame_equal(
        result.baseline.paths[["latitude", "longitude"]],
        result.changed.paths[["latitude", "longitude"]],
    )


def test_migration_tracker_interpolates_population_travelers_along_routes():
    frame = population_frame(years=1, animals=2)
    result = simulate_population_scenario(
        frame, WeatherScenario(), "aerial", activated_effects={}
    )
    start_deck = build_population_deck(result, progress=0.0)
    finish_deck = build_population_deck(result, progress=1.0)
    start_markers = start_deck.layers[-1].data
    finish_markers = finish_deck.layers[-1].data
    assert len(start_markers) == len(finish_markers) == 1
    assert start_markers[0]["name"] == "Spring baseline traveler"
    assert start_markers[0]["position"] != finish_markers[0]["position"]

    first_step = build_population_deck(result, progress=0.50).layers[-1].data[0]["position"]
    second_step = build_population_deck(result, progress=0.51).layers[-1].data[0]["position"]
    assert first_step != second_step
    assert abs(first_step[0] - second_step[0]) < 0.5
    assert abs(first_step[1] - second_step[1]) < 0.5


def test_animation_track_moves_at_every_progress_increment():
    route = pd.DataFrame(
        {
            "progress": [0.0, 0.1, 0.7, 1.0],
            "latitude": [10.0, 10.0, 20.0, 40.0],
            "longitude": [-100.0, -100.0, -90.0, -80.0],
            "season": ["spring migration"] * 4,
        }
    )

    track = build_animation_track(route)
    positions = list(zip(track.longitude, track.latitude))

    assert len(track) == 101
    assert np.allclose(track.progress, [index / 100 for index in range(101)])
    assert all(first != second for first, second in zip(positions, positions[1:]))


def test_playback_mode_keeps_routes_and_travelers_but_omits_static_recordings():
    frame = population_frame(years=2, animals=4)
    result = simulate_population_scenario(
        frame, WeatherScenario(), "aerial", activated_effects={}
    )

    deck = build_population_deck(result, progress=0.25, playback_mode=True)
    names = [record["name"] for layer in deck.layers for record in layer.data]

    assert "Recorded spring journey" not in names
    assert "Baseline population route" in names
    assert "Spring baseline traveler" in names


def test_scenario_run_has_four_season_colored_travelers():
    spring = population_frame(years=2, animals=6)
    spring.loc[spring.animal_id.isin(["bird-3", "bird-4", "bird-5"]), "longitude"] += 8
    fall = spring.copy()
    fall["season"] = "fall migration"
    fall["timestamp_utc"] = fall.timestamp_utc + pd.Timedelta(days=180)
    fall["latitude"] = fall.groupby(["animal_id", "year"]).latitude.transform(
        lambda values: values.iloc[::-1].to_numpy()
    )
    frame = pd.concat([spring, fall], ignore_index=True)
    result = simulate_population_scenario(
        frame,
        WeatherScenario(wind_speed_change_kmh=10),
        "aerial",
        activated_effects={
            "spring migration": {
                "corridor_choice": {
                    "kind": "classification",
                    "baseline": {"corridor 1": 0.5, "corridor 2": 0.5},
                    "scenario": {"corridor 1": 0.8, "corridor 2": 0.2},
                }
            }
        },
    )

    deck = build_population_deck(result, progress=0.35, playback_mode=True)
    travelers = {
        record["name"]: record
        for layer in deck.layers
        for record in layer.data
        if "traveler" in record["name"]
    }

    assert set(travelers) == {
        "Spring baseline traveler",
        "Fall baseline traveler",
        "Spring scenario traveler",
        "Fall scenario traveler",
    }
    assert travelers["Spring scenario traveler"]["color"][:3] == [164, 92, 255]
    assert travelers["Fall scenario traveler"]["color"][:3] == [255, 213, 74]
    assert "constrained scenario route" in travelers["Fall scenario traveler"]["emphasis"]
    assert set(result.activated_effects) == {"spring migration", "fall migration"}
    assert (
        result.activated_effects["fall migration"]["corridor_choice"]["support"]
        == "paired-season projection"
    )


def test_activated_scenario_draws_a_distinct_labeled_projected_route():
    frame = population_frame(years=3, animals=6)
    frame.loc[frame.animal_id.isin(["bird-3", "bird-4", "bird-5"]), "longitude"] += 8
    result = simulate_population_scenario(
        frame,
        WeatherScenario(wind_speed_change_kmh=10),
        "aerial",
        activated_effects={
            "spring migration": {
                "corridor_choice": {
                    "kind": "classification",
                    "baseline": {"corridor 1": 0.5, "corridor 2": 0.5},
                    "scenario": {"corridor 1": 0.8, "corridor 2": 0.2},
                }
            }
        },
    )

    deck = build_population_deck(
        result, progress=0.5, focus_seasons=["spring migration"]
    )
    records = [record for layer in deck.layers for record in layer.data]
    baseline_route = next(
        record for record in records if record["name"] == "Baseline population route"
    )
    scenario_route = next(
        record for record in records if record["name"] == "Scenario population route"
    )

    assert scenario_route["path"] != baseline_route["path"]
    assert len(scenario_route["path"]) == 36
    assert scenario_route["emphasis"] == (
        "Generated from seasonal evidence and anchored to recorded endpoints"
    )
    assert np.allclose(scenario_route["path"][0], baseline_route["path"][0])
    assert np.allclose(scenario_route["path"][-1], baseline_route["path"][-1])
    early_shift = np.linalg.norm(
        np.asarray(scenario_route["path"][1]) - np.asarray(baseline_route["path"][1])
    )
    middle = len(scenario_route["path"]) // 2
    middle_shift = np.linalg.norm(
        np.asarray(scenario_route["path"][middle])
        - np.asarray(baseline_route["path"][middle])
    )
    assert early_shift < middle_shift * 0.1
    assert {record["name"] for record in records} >= {
        "Spring baseline traveler",
        "Spring scenario traveler",
    }
    assert all(layer.pickable is False for layer in deck.layers)
    assert '"tooltip"' not in deck.to_json().lower()


def test_non_spatial_stopover_effect_draws_evidence_weighted_route_without_stopover_dots():
    frame = population_frame(years=3, animals=6)
    result = simulate_population_scenario(
        frame,
        WeatherScenario(tailwind_change_kmh=12),
        "aerial",
        activated_effects={
            "spring migration": {
                "stopovers": {
                    "kind": "continuous",
                    "baseline": 7.0,
                    "scenario": 5.0,
                    "delta": -2.0,
                    "unit": "stops",
                }
            }
        },
    )

    deck = build_population_deck(
        result, progress=0.5, focus_seasons=["spring migration"]
    )
    records = [record for layer in deck.layers for record in layer.data]
    names = [record["name"] for record in records]

    assert "Scenario population route" in names
    assert not any("rest stop" in name.lower() for name in names)
    assert {name for name in names if "traveler" in name} == {
        "Spring baseline traveler",
        "Spring scenario traveler",
    }


def test_blue_whale_centerline_uses_recorded_guatemala_and_southern_california_terminals():
    route = pd.DataFrame(
        {
            "progress": [0.0, 0.5, 1.0],
            "latitude": [23.0, 27.0, 30.0],
            "longitude": [-106.0, -112.0, -116.0],
        }
    )
    paths = pd.DataFrame(
        {
            "species": ["Balaenoptera musculus"],
            "population": ["Northeast Pacific"],
        }
    )

    spring = _extend_route_to_known_terminals(route, "spring migration", paths)
    fall = _extend_route_to_known_terminals(route, "fall migration", paths)

    assert np.allclose(spring.iloc[0][["latitude", "longitude"]], [12.124, -91.998])
    assert np.allclose(spring.iloc[-1][["latitude", "longitude"]], [34.141, -120.565])
    assert np.allclose(fall.iloc[0][["latitude", "longitude"]], [34.141, -120.565])
    assert np.allclose(fall.iloc[-1][["latitude", "longitude"]], [12.124, -91.998])


def test_continuous_scenario_is_scaled_to_both_recorded_seasons():
    spring = population_frame(years=2, animals=4)
    fall = spring.copy()
    fall["season"] = "fall migration"
    fall["timestamp_utc"] = fall.timestamp_utc + pd.Timedelta(days=180)
    fall.loc[fall.animal_id.eq("bird-3"), "timestamp_utc"] += pd.Timedelta(days=2)
    frame = pd.concat([spring, fall], ignore_index=True)

    result = simulate_population_scenario(
        frame,
        WeatherScenario(tailwind_change_kmh=12),
        "aerial",
        activated_effects={
            "spring migration": {
                "stopovers": {
                    "kind": "continuous",
                    "baseline": 4.0,
                    "scenario": 2.0,
                    "delta": -2.0,
                    "unit": "stops",
                }
            }
        },
    )

    assert set(result.activated_effects) == {"spring migration", "fall migration"}
    fall_effect = result.activated_effects["fall migration"]["stopovers"]
    assert fall_effect["support"] == "paired-season projection"
    assert fall_effect["evidence_season"] == "spring migration"
    assert fall_effect["scenario"] <= fall_effect["baseline"]
