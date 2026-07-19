from __future__ import annotations

import numpy as np


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(delta_lon / 2) ** 2
    )
    return 6371.0088 * 2 * np.arctan2(np.sqrt(value), np.sqrt(1 - value))
