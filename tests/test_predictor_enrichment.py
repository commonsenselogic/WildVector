from unittest.mock import patch

import pandas as pd
import pytest

from core.predictor_enrichment import (
    fetch_hourly_land_features,
    join_bylot_lemmings,
    merge_feature_frames,
)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def point(species="Cathartes aura"):
    return pd.DataFrame(
        {
            "species": [species],
            "season": ["spring migration"],
            "year": [2010],
            "journey_id": ["bird|2010|spring migration"],
            "window_name": ["route_early_7d"],
            "start_date": [pd.Timestamp("2010-04-01").date()],
            "end_date": [pd.Timestamp("2010-04-01").date()],
            "latitude": [40.0],
            "longitude": [-76.0],
            "route_bearing_degrees": [0.0],
        }
    )


def test_hourly_conditions_derive_daylight_uplift_and_route_wind():
    hours = pd.date_range("2010-04-01", periods=24, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    daylight = [0.0] * 8 + [500.0] * 10 + [0.0] * 6
    payload = {
        "hourly": {
            "time": hours,
            "shortwave_radiation": daylight,
            "direct_radiation": daylight,
            "diffuse_radiation": [value / 5 for value in daylight],
            "boundary_layer_height": [1000.0] * 24,
            "cloud_cover": [20.0] * 24,
            "cape": [100.0] * 24,
            "wind_speed_10m": [10.0] * 24,
            "wind_direction_10m": [180.0] * 24,
            "snow_depth": [0.0] * 24,
            "soil_temperature_0_to_7cm": [8.0] * 24,
        }
    }
    with patch("core.predictor_enrichment._get", return_value=Response(payload)) as request:
        result = fetch_hourly_land_features(point())
    assert result.thermal_uplift_proxy.iloc[0] > 0
    assert result.flight_hour_tailwind_mean.iloc[0] > 0
    assert result.boundary_layer_height_mean.iloc[0] == 1000.0
    assert request.call_args.kwargs["params"]["models"] == "era5"


def test_fox_snow_uses_era5_land():
    hours = pd.date_range("2010-04-01", periods=24, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    payload = {"hourly": {"time": hours, "snow_depth": [0.4] * 24}}
    with patch("core.predictor_enrichment._get", return_value=Response(payload)) as request:
        result = fetch_hourly_land_features(point("Vulpes lagopus"))
    assert result.snow_depth_mean.iloc[0] == pytest.approx(0.4)
    assert request.call_args.kwargs["params"]["models"] == "era5_land"


def test_bylot_lemmings_join_by_year_without_touching_other_species():
    lemmings = pd.DataFrame(
        {"year": [2010], "lemming_density_index": [4.2], "lemming_source": ["test"]}
    )
    fox = point("Vulpes lagopus")
    joined = join_bylot_lemmings(fox, lemmings)
    assert joined.lemming_density_index.iloc[0] == 4.2
    assert join_bylot_lemmings(point(), lemmings).empty


def test_feature_merge_is_keyed_and_preserves_base_rows():
    base = point()[["journey_id", "window_name"]].assign(temperature_2m_mean=5.0)
    features = point()[["journey_id", "window_name"]].assign(snow_depth_mean=0.3)
    merged = merge_feature_frames(base, [features])
    assert len(merged) == 1
    assert merged.snow_depth_mean.iloc[0] == 0.3
    assert merged.temperature_2m_mean.iloc[0] == 5.0
