import pandas as pd

from core.land_routing import land_only_segments, route_is_over_land, route_over_land


def test_modeled_vulture_route_detours_around_gulf_water():
    route = pd.DataFrame(
        {
            "progress": [0.0, 0.5, 1.0],
            "latitude": [30.0, 24.0, 18.0],
            "longitude": [-97.0, -90.0, -91.0],
        }
    )
    constrained = route_over_land(route)
    assert route_is_over_land(constrained)
    assert constrained.progress.iloc[0] == 0.0
    assert constrained.progress.iloc[-1] == 1.0


def test_recorded_vulture_connectors_split_before_crossing_water():
    route = pd.DataFrame(
        {
            "progress": [0.0, 0.25, 0.5, 0.75, 1.0],
            "latitude": [30.0, 29.0, 25.0, 20.0, 19.0],
            "longitude": [-98.0, -97.0, -90.0, -92.0, -91.0],
        }
    )
    segments = land_only_segments(route)
    assert segments
    assert all(route_is_over_land(segment) for segment in segments)
