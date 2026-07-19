import numpy as np
import pandas as pd

from core.environmental_history import build_journey_weather_points
from core.outcome_model import (
    extract_journey_outcomes,
    fit_outcome_models,
    predict_activated_effects,
)
from core.scenario import WeatherScenario
from tests.test_population import population_frame


def weather_linked_history(years=10):
    telemetry = population_frame(years=years, animals=6, points=14)
    pattern = np.array([-4, 3, -1, 5, -3, 2, -5, 4, 0, 6], dtype=float)[:years]
    for index, year in enumerate(sorted(telemetry.year.unique())):
        telemetry.loc[telemetry.year.eq(year), "timestamp_utc"] += pd.Timedelta(
            days=float(3 * pattern[index])
        )
    environment_rows = []
    for index, year in enumerate(sorted(telemetry.year.unique())):
        environment_rows.append(
            {
                "species": "Testus migratorius",
                "population": "Test flyway",
                "movement_type": "aerial",
                "season": "spring migration",
                "year": int(year),
                "temperature_2m": float(pattern[index]),
                "surface_pressure": 1000.0,
                "precipitation": 1.0,
                "wind_u_component_10m": np.nan,
                "wind_v_component_10m": np.nan,
                "sea_surface_temperature": np.nan,
                "marine_current_u": np.nan,
                "marine_current_v": np.nan,
            }
        )
    return telemetry, pd.DataFrame(environment_rows)


def test_extracts_measurable_journey_outcomes_and_corridors():
    telemetry = population_frame(years=2, animals=6, points=12)
    telemetry.loc[telemetry.animal_id.isin(["bird-3", "bird-4", "bird-5"]), "longitude"] += 8
    outcomes = extract_journey_outcomes(telemetry)
    assert {
        "departure_day", "arrival_day", "duration_days", "pace_km_day",
        "stopover_count", "corridor_choice",
    } <= set(outcomes)
    assert outcomes.corridor_choice.nunique() == 2
    assert outcomes.duration_days.gt(0).all()


def test_forward_weather_model_must_beat_both_baselines_before_activation():
    telemetry, environment = weather_linked_history()
    bundle, validations = fit_outcome_models(telemetry, environment)
    departure = next(row for row in validations if row.outcome == "departure_date")
    assert departure.folds == 7
    assert departure.environmental_error < departure.seasonal_error
    assert departure.environmental_error < departure.persistence_error
    assert departure.status == "active historical association"
    effects = predict_activated_effects(bundle, WeatherScenario(temperature_change_c=2))
    assert effects["departure_date"]["delta"] > 0
    assert predict_activated_effects(bundle, WeatherScenario()) == {}


def test_too_few_forward_folds_never_activates():
    telemetry, environment = weather_linked_history(years=5)
    _, validations = fit_outcome_models(telemetry, environment)
    assert validations
    assert all(row.folds < 3 for row in validations)
    assert all(row.status == "not activated" for row in validations)


def test_journey_weather_isolated_by_outcome_window_and_future_year():
    telemetry, _ = weather_linked_history()
    weather = build_journey_weather_points(telemetry)
    year_pattern = dict(zip(sorted(telemetry.year.unique()), [-4, 3, -1, 5, -3, 2, -5, 4, 0, 6]))
    weather["temperature_2m_mean"] = weather.year.map(year_pattern).astype(float)
    bundle, validations = fit_outcome_models(telemetry, weather)
    departure = next(row for row in validations if row.outcome == "departure_date")
    assert departure.folds == 7
    assert departure.status == "active historical association"
    assert departure.feature_columns
    assert all(column.startswith(("departure_30d__", "departure_7d__")) for column in departure.feature_columns)
    assert bundle["schema_version"] == 4
    assert bundle["source_trials"]
    assert bundle["retained_sources"]["departure_date"] == ["atmosphere_daily"]
    assert bundle["evidence_sources"]["departure_date"] == ["atmosphere_daily"]
    assert all(row["model_status"] == "included" for row in bundle["source_trials"])
    assert all("evidence_status" in row for row in bundle["source_trials"])
