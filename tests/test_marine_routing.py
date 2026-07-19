import pandas as pd

from core.marine_routing import route_is_over_water, route_over_water, water_only_segments


def test_modeled_marine_route_detours_around_land():
    # This connector cuts across Baja California when drawn as a straight line.
    route = pd.DataFrame(
        {
            "progress": [0.0, 1.0],
            "latitude": [24.0, 28.0],
            "longitude": [-116.0, -111.0],
        }
    )

    constrained = route_over_water(route)

    assert len(constrained) > 2
    assert route_is_over_water(constrained)


def test_recorded_marine_connectors_are_split_instead_of_drawn_over_land():
    route = pd.DataFrame(
        {
            "progress": [0.0, 0.33, 0.66, 1.0],
            "latitude": [23.0, 24.0, 28.0, 29.0],
            "longitude": [-117.0, -116.0, -111.0, -110.0],
        }
    )

    segments = water_only_segments(route)

    assert segments
    assert all(route_is_over_water(segment) for segment in segments)
