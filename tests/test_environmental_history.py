import pandas as pd

from core.environmental_history import build_journey_weather_points, sanitize_journey_environment
from tests.test_population import population_frame


def test_journey_weather_points_cover_six_biological_windows_per_journey():
    telemetry = population_frame(years=2, animals=3, points=14)
    points = build_journey_weather_points(telemetry)
    counts = points.groupby("journey_id").window_name.nunique()
    assert counts.eq(6).all()
    assert set(points.window_name) == {
        "departure_30d", "departure_7d", "route_early_7d", "route_middle_7d",
        "route_late_7d", "arrival_14d",
    }
    requested = points.set_index("window_name").days_requested
    assert requested["departure_30d"].eq(30).all()
    assert requested["departure_7d"].eq(7).all()
    assert requested["arrival_14d"].eq(14).all()
    route = points[points.window_name.str.startswith("route_")]
    assert pd.to_datetime(route.anchor_date).dt.year.ge(2000).all()


def test_environment_quality_control_rejects_ice_sheet_as_snowpack():
    frame = pd.DataFrame(
        {
            "snow_depth_mean": [0.8, 33.33],
            "sea_ice_concentration": [0.7, 2.55],
        }
    )
    result = sanitize_journey_environment(frame)
    assert result.snow_depth_mean.notna().tolist() == [True, False]
    assert result.sea_ice_concentration.notna().tolist() == [True, False]
