import numpy as np
import pandas as pd
import pytest

from core.environmental_history import build_sampling_points
from core.population_model import apply_population_response_model, fit_population_response_model
from core.scenario import WeatherScenario
from tests.test_population import population_frame


def test_environment_model_uses_held_out_years_and_baseline():
    telemetry = population_frame(years=4, animals=5, points=18)
    points = build_sampling_points(telemetry, bins=12)
    progress = points.progress_bin / 11
    points["temperature_2m"] = 5 + 18 * progress + (points.year - points.year.min())
    points["surface_pressure"] = 1000 + np.sin(progress * np.pi * 2) * 8
    points["precipitation"] = 1.0
    points["wind_u_component_10m"] = 4 + 3 * progress
    points["wind_v_component_10m"] = 8 + points.year - points.year.min()
    points["sea_surface_temperature"] = np.nan
    points["marine_current_u"] = np.nan
    points["marine_current_v"] = np.nan
    bundle, validation = fit_population_response_model(telemetry, points, bins=12)
    assert validation.years == 4
    assert validation.folds == 4
    assert validation.samples > 30
    assert "wind_v_component_10m" in bundle["features"]
    assert validation.baseline_rmse_km_day >= 0


def test_diagnostic_velocity_model_cannot_generate_map_coordinates():
    with pytest.raises(RuntimeError, match="diagnostic only"):
        apply_population_response_model(
            population_frame(years=1), pd.DataFrame(), {}, WeatherScenario(temperature_change_c=2)
        )
