from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeatherScenario:
    """Classroom controls expressed as changes from typical seasonal conditions."""

    wind_speed_change_kmh: float = 0.0
    wind_direction_deg: float | None = None
    temperature_change_c: float = 0.0
    pressure_trend_hpa_per_day: float = 0.0
    current_speed_change_kmh: float = 0.0
    current_direction_deg: float | None = None
    tailwind_change_kmh: float = 0.0
    uplift_change_fraction: float = 0.0
    sea_ice_concentration_change: float = 0.0
    sea_ice_distance_change_km: float = 0.0

    @property
    def is_typical(self) -> bool:
        return (
            self.wind_speed_change_kmh == 0
            and self.wind_direction_deg is None
            and self.temperature_change_c == 0
            and self.pressure_trend_hpa_per_day == 0
            and self.current_speed_change_kmh == 0
            and self.current_direction_deg is None
            and self.tailwind_change_kmh == 0
            and self.uplift_change_fraction == 0
            and self.sea_ice_concentration_change == 0
            and self.sea_ice_distance_change_km == 0
        )


def typical_route_direction(events: pd.DataFrame):
    ordered = events.sort_values("timestamp_utc")
    if len(ordered) < 2:
        return 0.0
    start, end = ordered.iloc[0], ordered.iloc[-1]
    lat1, lat2 = np.radians(start.latitude), np.radians(end.latitude)
    delta_lon = np.radians(end.longitude - start.longitude)
    bearing = np.degrees(
        np.arctan2(
            np.sin(delta_lon) * np.cos(lat2),
            np.cos(lat1) * np.sin(lat2)
            - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon),
        )
    )
    return float((bearing + 360) % 360)


def _advance(latitude, longitude, east_km, north_km):
    new_latitude = float(latitude) + float(north_km) / 110.574
    scale = max(111.320 * math.cos(math.radians(float(latitude))), 0.01)
    new_longitude = ((float(longitude) + float(east_km) / scale + 180) % 360) - 180
    return new_latitude, new_longitude
