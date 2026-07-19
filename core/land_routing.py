from __future__ import annotations

from functools import lru_cache
import heapq
import math

import numpy as np
import pandas as pd
from global_land_mask import globe

from .forecaster import haversine


GRID_DEGREES = 0.18
_ROUTE_CACHE: dict[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...]] = {}


@lru_cache(maxsize=250_000)
def _is_land(latitude: float, longitude: float) -> bool:
    latitude = max(-89.99, min(89.99, float(latitude)))
    longitude = ((float(longitude) + 180.0) % 360.0) - 180.0
    return not bool(globe.is_ocean(latitude, longitude))


def _is_stable_land(latitude: float, longitude: float) -> bool:
    """Reject tiny coastal cells and islands that cannot join the mainland grid."""
    if not _is_land(round(latitude, 4), round(longitude, 4)):
        return False
    neighbors = 0
    for latitude_offset, longitude_offset in (
        (-0.12, -0.12), (-0.12, 0.0), (-0.12, 0.12),
        (0.0, -0.12), (0.0, 0.12),
        (0.12, -0.12), (0.12, 0.0), (0.12, 0.12),
    ):
        neighbors += _is_land(
            round(latitude + latitude_offset, 4),
            round(longitude + longitude_offset, 4),
        )
    return neighbors >= 4


def _segment_is_land(start: tuple[float, float], end: tuple[float, float]) -> bool:
    lat1, lon1 = start
    lat2, lon2 = end
    lon2 = lon1 + ((lon2 - lon1 + 180.0) % 360.0) - 180.0
    samples = max(3, int(max(abs(lat2 - lat1), abs(lon2 - lon1)) / 0.045) + 1)
    return all(
        _is_land(
            round(lat1 + (lat2 - lat1) * fraction, 4),
            round(lon1 + (lon2 - lon1) * fraction, 4),
        )
        for fraction in np.linspace(0.0, 1.0, samples)
    )


def _snap_to_land(point: tuple[float, float]) -> tuple[float, float]:
    if _is_stable_land(point[0], point[1]):
        return point
    latitude, longitude = point
    for ring in range(1, 36):
        radius = ring * GRID_DEGREES
        candidates = []
        for step in range(max(24, ring * 10)):
            angle = 2 * math.pi * step / max(24, ring * 10)
            candidate = (
                latitude + radius * math.sin(angle),
                longitude + radius * math.cos(angle) / max(math.cos(math.radians(latitude)), 0.2),
            )
            if _is_stable_land(candidate[0], candidate[1]):
                candidates.append(candidate)
        if candidates:
            return min(candidates, key=lambda p: (p[0] - latitude) ** 2 + (p[1] - longitude) ** 2)
    raise ValueError(f"No land cell found near {latitude:.3f}, {longitude:.3f}")


def _short_land_detour(start: tuple[float, float], end: tuple[float, float]):
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    for ring in range(1, 25):
        radius = ring * 0.04
        candidates = []
        for step in range(40):
            angle = 2 * math.pi * step / 40
            candidate = (
                midpoint[0] + radius * math.sin(angle),
                midpoint[1] + radius * math.cos(angle) / max(math.cos(math.radians(midpoint[0])), 0.2),
            )
            if (
                _is_stable_land(candidate[0], candidate[1])
                and _segment_is_land(start, candidate)
                and _segment_is_land(candidate, end)
            ):
                candidates.append(candidate)
        if candidates:
            waypoint = min(candidates, key=lambda point: math.dist(start, point) + math.dist(point, end))
            return [start, waypoint, end]
    return None


def _grid_route(start: tuple[float, float], end: tuple[float, float], padding: float):
    start, end = _snap_to_land(start), _snap_to_land(end)
    if _segment_is_land(start, end):
        return [start, end]
    short_detour = _short_land_detour(start, end)
    if short_detour is not None:
        return short_detour
    min_lat, max_lat = min(start[0], end[0]) - padding, max(start[0], end[0]) + padding
    end_lon = start[1] + ((end[1] - start[1] + 180.0) % 360.0) - 180.0
    min_lon, max_lon = min(start[1], end_lon) - padding, max(start[1], end_lon) + padding
    rows = int(math.ceil((max_lat - min_lat) / GRID_DEGREES)) + 1
    columns = int(math.ceil((max_lon - min_lon) / GRID_DEGREES)) + 1

    def coordinate(node):
        return min_lat + node[0] * GRID_DEGREES, min_lon + node[1] * GRID_DEGREES

    def nearest(point):
        origin = (
            int(round((point[0] - min_lat) / GRID_DEGREES)),
            int(round((point[1] - min_lon) / GRID_DEGREES)),
        )
        for radius in range(24):
            candidates = []
            for row_offset in range(-radius, radius + 1):
                for column_offset in range(-radius, radius + 1):
                    node = origin[0] + row_offset, origin[1] + column_offset
                    if not (0 <= node[0] < rows and 0 <= node[1] < columns):
                        continue
                    location = coordinate(node)
                    if _is_land(round(location[0], 4), round(location[1], 4)) and _segment_is_land(point, location):
                        candidates.append(node)
            if candidates:
                return min(candidates, key=lambda node: math.dist(coordinate(node), point))
        raise ValueError("No connected land grid node found near a route endpoint.")

    start_node, end_node = nearest(start), nearest((end[0], end_lon))
    frontier = [(0.0, start_node)]
    cost = {start_node: 0.0}
    came_from = {}
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == end_node:
            break
        for row_offset, column_offset in directions:
            neighbor = current[0] + row_offset, current[1] + column_offset
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < columns):
                continue
            location = coordinate(neighbor)
            if not _is_land(round(location[0], 4), round(location[1], 4)):
                continue
            if not _segment_is_land(coordinate(current), location):
                continue
            next_cost = cost[current] + math.hypot(row_offset, column_offset)
            if next_cost >= cost.get(neighbor, float("inf")):
                continue
            cost[neighbor] = next_cost
            came_from[neighbor] = current
            heuristic = math.hypot(neighbor[0] - end_node[0], neighbor[1] - end_node[1])
            heapq.heappush(frontier, (next_cost + heuristic, neighbor))
    if end_node not in cost:
        raise ValueError("No land-only connection found inside the routing grid.")
    nodes = [end_node]
    while nodes[-1] != start_node:
        nodes.append(came_from[nodes[-1]])
    nodes.reverse()
    raw = [start] + [coordinate(node) for node in nodes[1:-1]] + [end]
    simplified = [raw[0]]
    cursor = 0
    while cursor < len(raw) - 1:
        target = len(raw) - 1
        while target > cursor + 1 and not _segment_is_land(raw[cursor], raw[target]):
            target -= 1
        simplified.append(raw[target])
        cursor = target
    return simplified


def route_over_land(route: pd.DataFrame) -> pd.DataFrame:
    """Snap a generated vulture centerline to land and route around water."""
    if route.empty:
        return route.copy()
    ordered = route.sort_values("progress")
    cache_key = tuple(
        (round(float(row.progress), 6), round(float(row.latitude), 6), round(float(row.longitude), 6))
        for row in ordered.itertuples()
    )
    if cache_key in _ROUTE_CACHE:
        return pd.DataFrame(_ROUTE_CACHE[cache_key], columns=["progress", "latitude", "longitude"])
    source = [_snap_to_land((float(row.latitude), float(row.longitude))) for row in ordered.itertuples()]
    routed = [source[0]]
    for start, end in zip(source, source[1:]):
        if _segment_is_land(start, end):
            segment = [start, end]
        else:
            try:
                segment = _grid_route(start, end, 2.0)
            except ValueError:
                segment = _grid_route(start, end, 8.0)
        routed.extend(segment[1:])
    safe = [routed[0]]
    for point in routed[1:]:
        if _segment_is_land(safe[-1], point):
            safe.append(point)
            continue
        try:
            replacement = _grid_route(safe[-1], point, 2.0)
        except ValueError:
            replacement = _grid_route(safe[-1], point, 8.0)
        safe.extend(replacement[1:])
    routed = safe
    repaired = [routed[0]]
    for point in routed[1:]:
        if _segment_is_land(repaired[-1], point):
            repaired.append(point)
            continue
        detour = _short_land_detour(repaired[-1], point)
        if detour is None:
            detour = _grid_route(repaired[-1], point, 8.0)
        repaired.extend(detour[1:])
    routed = repaired
    coordinates = pd.DataFrame(routed, columns=["latitude", "longitude"])
    distance = haversine(
        coordinates.latitude.shift(), coordinates.longitude.shift(),
        coordinates.latitude, coordinates.longitude,
    ).fillna(0.0)
    cumulative = distance.cumsum()
    coordinates["progress"] = cumulative / cumulative.iloc[-1] if cumulative.iloc[-1] > 0 else 0.0
    result = coordinates[["progress", "latitude", "longitude"]]
    _ROUTE_CACHE[cache_key] = tuple(tuple(row) for row in result.itertuples(index=False, name=None))
    return result.copy()


def route_is_over_land(route: pd.DataFrame) -> bool:
    points = list(zip(route.latitude.astype(float), route.longitude.astype(float)))
    return bool(points) and all(_segment_is_land(start, end) for start, end in zip(points, points[1:]))


def land_only_segments(route: pd.DataFrame) -> list[pd.DataFrame]:
    """Split recorded fixes so the display never draws an unobserved water connector."""
    ordered = route.sort_values("progress").reset_index(drop=True)
    segments, current = [], []
    for _, point in ordered.iterrows():
        location = float(point.latitude), float(point.longitude)
        if not _is_land(round(location[0], 4), round(location[1], 4)):
            if len(current) >= 2:
                segments.append(pd.DataFrame(current))
            current = []
            continue
        if current:
            previous = float(current[-1]["latitude"]), float(current[-1]["longitude"])
            if not _segment_is_land(previous, location):
                if len(current) >= 2:
                    segments.append(pd.DataFrame(current))
                current = []
        current.append(point.to_dict())
    if len(current) >= 2:
        segments.append(pd.DataFrame(current))
    return segments
