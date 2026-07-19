from __future__ import annotations

from functools import lru_cache
import heapq
import math

import numpy as np
import pandas as pd
from global_land_mask import globe

from .forecaster import haversine


GRID_DEGREES = 0.16
_ROUTE_CACHE: dict[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...]] = {}


@lru_cache(maxsize=250_000)
def _is_ocean(latitude: float, longitude: float) -> bool:
    latitude = max(-89.99, min(89.99, float(latitude)))
    longitude = ((float(longitude) + 180.0) % 360.0) - 180.0
    return bool(globe.is_ocean(latitude, longitude))


def _segment_is_ocean(start: tuple[float, float], end: tuple[float, float]) -> bool:
    lat1, lon1 = start
    lat2, lon2 = end
    lon2 = lon1 + ((lon2 - lon1 + 180.0) % 360.0) - 180.0
    samples = max(3, int(max(abs(lat2 - lat1), abs(lon2 - lon1)) / 0.045) + 1)
    for fraction in np.linspace(0.0, 1.0, samples):
        if not _is_ocean(
            round(lat1 + (lat2 - lat1) * fraction, 4),
            round(lon1 + (lon2 - lon1) * fraction, 4),
        ):
            return False
    return True


def _snap_to_ocean(point: tuple[float, float]) -> tuple[float, float]:
    latitude, longitude = point
    if _is_ocean(round(latitude, 4), round(longitude, 4)):
        return float(latitude), float(longitude)
    for ring in range(1, 31):
        radius = ring * GRID_DEGREES
        candidates = []
        for step in range(max(16, ring * 8)):
            angle = 2 * math.pi * step / max(16, ring * 8)
            candidate = (
                latitude + radius * math.sin(angle),
                longitude + radius * math.cos(angle) / max(math.cos(math.radians(latitude)), 0.2),
            )
            if _is_ocean(round(candidate[0], 4), round(candidate[1], 4)):
                candidates.append(candidate)
        if candidates:
            return min(
                candidates,
                key=lambda candidate: (candidate[0] - latitude) ** 2
                + (candidate[1] - longitude) ** 2,
            )
    raise ValueError(f"No ocean cell found near {latitude:.3f}, {longitude:.3f}")


def _short_ocean_detour(
    start: tuple[float, float], end: tuple[float, float]
) -> list[tuple[float, float]] | None:
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    candidates = []
    for ring in range(1, 18):
        radius = ring * 0.06
        for step in range(32):
            angle = 2 * math.pi * step / 32
            candidate = (
                midpoint[0] + radius * math.sin(angle),
                midpoint[1] + radius * math.cos(angle)
                / max(math.cos(math.radians(midpoint[0])), 0.2),
            )
            if (
                _is_ocean(round(candidate[0], 4), round(candidate[1], 4))
                and _segment_is_ocean(start, candidate)
                and _segment_is_ocean(candidate, end)
            ):
                candidates.append(candidate)
        if candidates:
            waypoint = min(
                candidates,
                key=lambda point: math.hypot(point[0] - start[0], point[1] - start[1])
                + math.hypot(end[0] - point[0], end[1] - point[1]),
            )
            return [start, waypoint, end]
    return None


def _grid_route(
    start: tuple[float, float], end: tuple[float, float], padding: float
) -> list[tuple[float, float]]:
    start = _snap_to_ocean(start)
    end = _snap_to_ocean(end)
    if _segment_is_ocean(start, end):
        return [start, end]
    short_detour = _short_ocean_detour(start, end)
    if short_detour is not None:
        return short_detour

    start_lat, start_lon = start
    end_lat, end_lon = end
    end_lon = start_lon + ((end_lon - start_lon + 180.0) % 360.0) - 180.0
    minimum_latitude = min(start_lat, end_lat) - padding
    maximum_latitude = max(start_lat, end_lat) + padding
    minimum_longitude = min(start_lon, end_lon) - padding
    maximum_longitude = max(start_lon, end_lon) + padding
    rows = int(math.ceil((maximum_latitude - minimum_latitude) / GRID_DEGREES)) + 1
    columns = int(math.ceil((maximum_longitude - minimum_longitude) / GRID_DEGREES)) + 1

    def coordinate(node: tuple[int, int]) -> tuple[float, float]:
        row, column = node
        return (
            minimum_latitude + row * GRID_DEGREES,
            minimum_longitude + column * GRID_DEGREES,
        )

    def nearest_node(point: tuple[float, float]) -> tuple[int, int]:
        row = int(round((point[0] - minimum_latitude) / GRID_DEGREES))
        column = int(round((point[1] - minimum_longitude) / GRID_DEGREES))
        candidates = []
        for radius in range(0, 12):
            for row_offset in range(-radius, radius + 1):
                for column_offset in range(-radius, radius + 1):
                    candidate = (row + row_offset, column + column_offset)
                    if not (0 <= candidate[0] < rows and 0 <= candidate[1] < columns):
                        continue
                    latitude, longitude = coordinate(candidate)
                    if _is_ocean(round(latitude, 4), round(longitude, 4)) and _segment_is_ocean(
                        point, (latitude, longitude)
                    ):
                        candidates.append(candidate)
            if candidates:
                return min(
                    candidates,
                    key=lambda node: (coordinate(node)[0] - point[0]) ** 2
                    + (coordinate(node)[1] - point[1]) ** 2,
                )
        raise ValueError("No ocean grid node found near a route endpoint.")

    start_node, end_node = nearest_node(start), nearest_node((end_lat, end_lon))
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start_node)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    costs = {start_node: 0.0}
    directions = (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    )
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == end_node:
            break
        for row_offset, column_offset in directions:
            neighbor = (current[0] + row_offset, current[1] + column_offset)
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < columns):
                continue
            latitude, longitude = coordinate(neighbor)
            if not _is_ocean(round(latitude, 4), round(longitude, 4)):
                continue
            if not _segment_is_ocean(coordinate(current), (latitude, longitude)):
                continue
            step_cost = math.hypot(row_offset, column_offset)
            new_cost = costs[current] + step_cost
            if new_cost >= costs.get(neighbor, float("inf")):
                continue
            costs[neighbor] = new_cost
            came_from[neighbor] = current
            heuristic = math.hypot(neighbor[0] - end_node[0], neighbor[1] - end_node[1])
            heapq.heappush(frontier, (new_cost + heuristic, neighbor))
    if end_node not in costs:
        raise ValueError("No water-only connection found inside the routing grid.")

    nodes = [end_node]
    while nodes[-1] != start_node:
        nodes.append(came_from[nodes[-1]])
    nodes.reverse()
    raw = [start] + [coordinate(node) for node in nodes[1:-1]] + [end]

    simplified = [raw[0]]
    cursor = 0
    while cursor < len(raw) - 1:
        target = len(raw) - 1
        while target > cursor + 1 and not _segment_is_ocean(raw[cursor], raw[target]):
            target -= 1
        simplified.append(raw[target])
        cursor = target
    return simplified


def route_over_water(route: pd.DataFrame) -> pd.DataFrame:
    """Snap a modeled marine centerline to water and route connectors around land."""
    if route.empty:
        return route.copy()
    ordered = route.sort_values("progress")
    cache_key = tuple(
        (round(float(row.progress), 6), round(float(row.latitude), 6), round(float(row.longitude), 6))
        for row in ordered.itertuples()
    )
    cached = _ROUTE_CACHE.get(cache_key)
    if cached is not None:
        return pd.DataFrame(cached, columns=["progress", "latitude", "longitude"])
    source = [
        _snap_to_ocean((float(row.latitude), float(row.longitude)))
        for row in ordered.itertuples()
    ]
    routed = [source[0]]
    for start, end in zip(source, source[1:]):
        if _segment_is_ocean(start, end):
            segment = [start, end]
        else:
            try:
                segment = _grid_route(start, end, padding=1.2)
            except ValueError:
                # Long eastern-Pacific legs may need to clear the full Baja
                # peninsula before reconnecting with an observed offshore fix.
                segment = _grid_route(start, end, padding=8.0)
        routed.extend(segment[1:])

    safe = [routed[0]]
    for point in routed[1:]:
        if _segment_is_ocean(safe[-1], point):
            safe.append(point)
            continue
        try:
            replacement = _grid_route(safe[-1], point, padding=1.2)
        except ValueError:
            replacement = _grid_route(safe[-1], point, padding=8.0)
        safe.extend(replacement[1:])
    routed = safe

    coordinates = pd.DataFrame(routed, columns=["latitude", "longitude"])
    if len(coordinates) == 1:
        coordinates["progress"] = 0.0
        return coordinates
    distance = haversine(
        coordinates.latitude.shift(), coordinates.longitude.shift(),
        coordinates.latitude, coordinates.longitude,
    ).fillna(0.0)
    cumulative = distance.cumsum()
    coordinates["progress"] = (
        cumulative / cumulative.iloc[-1]
        if cumulative.iloc[-1] > 0
        else np.linspace(0.0, 1.0, len(coordinates))
    )
    result = coordinates[["progress", "latitude", "longitude"]]
    if len(_ROUTE_CACHE) >= 128:
        _ROUTE_CACHE.pop(next(iter(_ROUTE_CACHE)))
    _ROUTE_CACHE[cache_key] = tuple(
        (float(row.progress), float(row.latitude), float(row.longitude))
        for row in result.itertuples()
    )
    return result


def route_is_over_water(route: pd.DataFrame) -> bool:
    points = list(zip(route.latitude.astype(float), route.longitude.astype(float)))
    return all(_segment_is_ocean(start, end) for start, end in zip(points, points[1:]))


def water_only_segments(route: pd.DataFrame) -> list[pd.DataFrame]:
    """Split observed marine fixes wherever a straight connector crosses land."""
    if route.empty:
        return []
    ordered = route.sort_values("progress").reset_index(drop=True)
    starts = [0]
    breaks = []
    points = list(zip(ordered.latitude.astype(float), ordered.longitude.astype(float)))
    for index, (start, end) in enumerate(zip(points, points[1:]), start=1):
        if not _segment_is_ocean(start, end):
            breaks.append(index)
    segments = []
    for stop in breaks + [len(ordered)]:
        start = starts[-1]
        if stop - start >= 2:
            segments.append(ordered.iloc[start:stop].copy())
        starts.append(stop)
    return segments
