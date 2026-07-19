from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

from .land_routing import land_only_segments, route_over_land
from .marine_routing import route_over_water, water_only_segments
from .population import build_projected_route


SEASON_COLORS = {
    "spring migration": [65, 145, 205, 70],
    "fall migration": [45, 165, 120, 70],
}
SCENARIO_COLORS = {
    "spring migration": [164, 92, 255, 255],
    "fall migration": [255, 213, 74, 255],
}
KNOWN_RANGE_TERMINALS = {
    ("Balaenoptera musculus", "Northeast Pacific"): {
        # Recorded fixes near the Pacific coast of Guatemala and the Southern
        # California Bight, used as the full population-range terminals.
        "south": (12.124, -91.998),
        "north": (34.141, -120.565),
    }
}
BLUE_WHALE_PACIFIC_CORRIDOR = [
    (12.124, -91.998),
    (14.5, -96.5),
    (18.0, -104.5),
    (22.0, -111.0),
    (27.0, -116.0),
    (31.0, -118.5),
    (34.141, -120.565),
]


def build_animation_track(route, increments: int = 100):
    """Resample a route into one distance-spaced position per playback increment."""
    if increments < 1:
        raise ValueError("Animation increments must be positive.")
    ordered = route.sort_values("progress")
    latitudes = ordered.latitude.to_numpy(float)
    longitudes = np.degrees(np.unwrap(np.radians(ordered.longitude.to_numpy(float))))
    latitude_delta = np.diff(latitudes)
    longitude_delta = np.diff(longitudes) * np.cos(
        np.radians((latitudes[:-1] + latitudes[1:]) / 2.0)
    )
    segment_distance = np.hypot(latitude_delta, longitude_delta)
    keep = np.r_[True, segment_distance > 1e-9]
    latitudes = latitudes[keep]
    longitudes = longitudes[keep]
    if len(latitudes) < 2:
        raise ValueError("An animation route needs at least two distinct positions.")
    latitude_delta = np.diff(latitudes)
    longitude_delta = np.diff(longitudes) * np.cos(
        np.radians((latitudes[:-1] + latitudes[1:]) / 2.0)
    )
    distance = np.r_[0.0, np.cumsum(np.hypot(latitude_delta, longitude_delta))]
    targets = np.linspace(0.0, distance[-1], increments + 1)
    track = ordered.iloc[:0].reindex(range(increments + 1)).copy()
    track["progress"] = np.linspace(0.0, 1.0, increments + 1)
    track["latitude"] = np.interp(targets, distance, latitudes)
    track["longitude"] = (
        (np.interp(targets, distance, longitudes) + 180.0) % 360.0
    ) - 180.0
    for column in ("season", "movement_type"):
        if column in ordered:
            track[column] = ordered[column].iloc[0]
    return track


def _interpolate_from_track(track, progress: float) -> list[float]:
    index = int(round(float(np.clip(progress, 0.0, 1.0)) * (len(track) - 1)))
    point = track.iloc[index]
    return [float(point.longitude), float(point.latitude)]


def _route_marker(track, progress: float, *, name: str, color: list[int], evidence: str):
    point = track.iloc[0]
    return {
        "name": name,
        "journey": "Population projection",
        "season": str(point.season).title(),
        "year": "Multiple years",
        "position": _interpolate_from_track(track, progress),
        "color": color,
        "progress_label": f"{progress:.0%} through migration",
        "emphasis": evidence,
    }


def _extend_route_to_known_terminals(route, season: str, paths):
    """Extend a population centerline through known, water-safe terminals."""
    if not {"species", "population"}.issubset(paths):
        return route
    key = (str(paths.species.iloc[0]), str(paths.population.iloc[0]))
    terminals = KNOWN_RANGE_TERMINALS.get(key)
    if terminals is None:
        return route
    extended = route.sort_values("progress").copy().reset_index(drop=True)
    progress = extended.progress.to_numpy(float)
    corridor = BLUE_WHALE_PACIFIC_CORRIDOR
    if not str(season).startswith("spring"):
        corridor = list(reversed(corridor))
    latitude = np.asarray([point[0] for point in corridor], dtype=float)
    longitude = np.asarray([point[1] for point in corridor], dtype=float)
    segment_distance = np.hypot(
        np.diff(latitude),
        np.diff(longitude) * np.cos(np.radians((latitude[:-1] + latitude[1:]) / 2.0)),
    )
    corridor_progress = np.r_[0.0, np.cumsum(segment_distance)]
    corridor_progress /= corridor_progress[-1]
    extended["latitude"] = np.interp(progress, corridor_progress, latitude)
    extended["longitude"] = np.interp(progress, corridor_progress, longitude)
    start, end = corridor[0], corridor[-1]
    extended.loc[extended.index[0], ["latitude", "longitude"]] = start
    extended.loc[extended.index[-1], ["latitude", "longitude"]] = end
    return extended


def _anchor_route_endpoints(route, baseline_route, max_shift_degrees: float | None = None):
    """Keep a scenario inside the known seasonal journey's start/end anchors."""
    anchored = route.sort_values("progress").copy().reset_index(drop=True)
    baseline = baseline_route.sort_values("progress")
    progress = anchored.progress.to_numpy(float)
    baseline_progress = baseline.progress.to_numpy(float)
    baseline_latitude = np.interp(
        progress, baseline_progress, baseline.latitude.to_numpy(float)
    )
    baseline_longitude = np.interp(
        progress,
        baseline_progress,
        np.degrees(np.unwrap(np.radians(baseline.longitude.to_numpy(float)))),
    )
    scenario_longitude = np.degrees(
        np.unwrap(np.radians(anchored.longitude.to_numpy(float)))
    )
    start_latitude_shift = float(anchored.latitude.iloc[0] - baseline_latitude[0])
    end_latitude_shift = float(anchored.latitude.iloc[-1] - baseline_latitude[-1])
    start_longitude_shift = float(scenario_longitude[0] - baseline_longitude[0])
    end_longitude_shift = float(scenario_longitude[-1] - baseline_longitude[-1])
    anchored["latitude"] = anchored.latitude.to_numpy(float) - (
        (1.0 - progress) * start_latitude_shift + progress * end_latitude_shift
    )
    anchored["longitude"] = (
        scenario_longitude
        - ((1.0 - progress) * start_longitude_shift + progress * end_longitude_shift)
        + 180.0
    ) % 360.0 - 180.0
    # A single coincident endpoint can still look unanchored when the next route
    # vertex immediately jumps away. Smoothly taper the entire deformation near
    # departure and arrival so whale corridors visibly share their terminal legs.
    envelope = np.sin(np.pi * np.clip(progress, 0.0, 1.0)) ** 2
    corrected_longitude = np.degrees(
        np.unwrap(np.radians(anchored.longitude.to_numpy(float)))
    )
    latitude_shift = (anchored.latitude.to_numpy(float) - baseline_latitude) * envelope
    longitude_shift = (corrected_longitude - baseline_longitude) * envelope
    if max_shift_degrees is not None:
        shift_size = np.hypot(
            latitude_shift,
            longitude_shift * np.cos(np.radians(baseline_latitude)),
        )
        scale = np.minimum(1.0, max_shift_degrees / np.maximum(shift_size, 1e-9))
        latitude_shift *= scale
        longitude_shift *= scale
    anchored["latitude"] = baseline_latitude + latitude_shift
    anchored["longitude"] = (
        baseline_longitude + longitude_shift + 180.0
    ) % 360.0 - 180.0
    anchored.loc[anchored.index[0], ["latitude", "longitude"]] = [
        baseline_latitude[0],
        (baseline_longitude[0] + 180.0) % 360.0 - 180.0,
    ]
    anchored.loc[anchored.index[-1], ["latitude", "longitude"]] = [
        baseline_latitude[-1],
        (baseline_longitude[-1] + 180.0) % 360.0 - 180.0,
    ]
    return anchored


def _move_deformation_to_display_corridor(route, source_baseline, display_baseline):
    """Apply the modeled source-route change to the staged display corridor."""
    changed = route.sort_values("progress").copy().reset_index(drop=True)
    progress = changed.progress.to_numpy(float)
    source = source_baseline.sort_values("progress")
    display = display_baseline.sort_values("progress")
    source_latitude = np.interp(progress, source.progress, source.latitude)
    display_latitude = np.interp(progress, display.progress, display.latitude)
    source_longitude = np.interp(
        progress,
        source.progress,
        np.degrees(np.unwrap(np.radians(source.longitude.to_numpy(float)))),
    )
    display_longitude = np.interp(
        progress,
        display.progress,
        np.degrees(np.unwrap(np.radians(display.longitude.to_numpy(float)))),
    )
    changed_longitude = np.degrees(
        np.unwrap(np.radians(changed.longitude.to_numpy(float)))
    )
    changed["latitude"] = display_latitude + (
        changed.latitude.to_numpy(float) - source_latitude
    )
    changed["longitude"] = (
        display_longitude + changed_longitude - source_longitude + 180.0
    ) % 360.0 - 180.0
    return changed


def _mean_route_shift_km(route, baseline_route) -> float:
    progress = route.progress.to_numpy(float)
    baseline = baseline_route.sort_values("progress")
    baseline_latitude = np.interp(
        progress, baseline.progress, baseline.latitude.to_numpy(float)
    )
    baseline_longitude = np.interp(
        progress,
        baseline.progress,
        np.degrees(np.unwrap(np.radians(baseline.longitude.to_numpy(float)))),
    )
    route_longitude = np.degrees(np.unwrap(np.radians(route.longitude.to_numpy(float))))
    latitude_km = (route.latitude.to_numpy(float) - baseline_latitude) * 111.0
    longitude_km = (route_longitude - baseline_longitude) * 111.0 * np.cos(
        np.radians((route.latitude.to_numpy(float) + baseline_latitude) / 2.0)
    )
    return float(np.mean(np.hypot(latitude_km, longitude_km)))


def _transfer_route_deformation(target_baseline, donor_baseline, donor_scenario):
    """Apply a reciprocal season's evidence-weighted shape to a known seasonal path."""
    target = target_baseline.sort_values("progress").copy().reset_index(drop=True)
    donor_progress = donor_baseline.progress.to_numpy(float)
    source_progress = 1.0 - target.progress.to_numpy(float)
    donor_baseline_latitude = np.interp(
        source_progress, donor_progress, donor_baseline.latitude.to_numpy(float)
    )
    donor_scenario_latitude = np.interp(
        source_progress, donor_scenario.progress, donor_scenario.latitude.to_numpy(float)
    )
    donor_baseline_longitude = np.interp(
        source_progress,
        donor_progress,
        np.degrees(np.unwrap(np.radians(donor_baseline.longitude.to_numpy(float)))),
    )
    donor_scenario_longitude = np.interp(
        source_progress,
        donor_scenario.progress,
        np.degrees(np.unwrap(np.radians(donor_scenario.longitude.to_numpy(float)))),
    )
    target["latitude"] = (
        target.latitude.to_numpy(float)
        + donor_scenario_latitude
        - donor_baseline_latitude
    )
    target["longitude"] = (
        np.degrees(np.unwrap(np.radians(target.longitude.to_numpy(float))))
        + donor_scenario_longitude
        - donor_baseline_longitude
        + 180.0
    ) % 360.0 - 180.0
    return _anchor_route_endpoints(target, target_baseline)


_ROUTE_GEOMETRY_CACHE: "OrderedDict" = OrderedDict()
_ROUTE_GEOMETRY_CACHE_SIZE = 16


def _route_geometry(population_scenario, focus_seasons=None, scenario_selected=True):
    """Build (and cache) the parts of the deck that don't depend on playback progress.

    Routing (land/water pathfinding, endpoint anchoring, deformation transfer) and the
    100-point animation resample are the expensive steps in the migration player, and
    none of them change between playback ticks -- only the traveler's progress does.
    Keying the cache on object identity means a single playback session (which reuses
    the same `population_scenario` across every ~180ms fragment tick) hits the cache on
    every tick after the first, instead of re-running the full routing pipeline ~5x/sec.
    """
    key = (id(population_scenario), tuple(sorted(focus_seasons or ())), bool(scenario_selected))
    cached = _ROUTE_GEOMETRY_CACHE.get(key)
    if cached is not None and cached[0] is population_scenario:
        _ROUTE_GEOMETRY_CACHE.move_to_end(key)
        return cached[1]

    geometry = _build_route_geometry(population_scenario, focus_seasons, scenario_selected)
    _ROUTE_GEOMETRY_CACHE[key] = (population_scenario, geometry)
    _ROUTE_GEOMETRY_CACHE.move_to_end(key)
    while len(_ROUTE_GEOMETRY_CACHE) > _ROUTE_GEOMETRY_CACHE_SIZE:
        _ROUTE_GEOMETRY_CACHE.popitem(last=False)
    return geometry


def _build_route_geometry(population_scenario, focus_seasons=None, scenario_selected=True):
    baseline = population_scenario.baseline
    changed = population_scenario.changed
    focus = set(focus_seasons or [])
    baseline_paths = baseline.paths
    changed_paths = changed.paths
    if focus:
        baseline_paths = baseline_paths[baseline_paths.season.isin(focus)]
        changed_paths = changed_paths[changed_paths.season.isin(focus)]
    is_marine = (
        "movement_type" in baseline_paths
        and baseline_paths.movement_type.astype(str).eq("marine").all()
    )
    is_vulture = (
        "species" in baseline_paths
        and baseline_paths.species.astype(str).eq("Cathartes aura").all()
    )

    observed_records = []
    for journey_id, group in baseline_paths.groupby("journey_id", sort=True):
        season = str(group.season.iloc[0])
        display_segments = (
            water_only_segments(group)
            if is_marine
            else land_only_segments(group) if is_vulture else [group]
        )
        for segment in display_segments:
            observed_records.append(
                {
                    "name": "Recorded spring journey" if season.startswith("spring") else "Recorded fall journey",
                    "journey": journey_id,
                    "season": season.title(),
                    "year": int(group.year.iloc[0]),
                    "path": segment.sort_values("progress")[["longitude", "latitude"]].values.tolist(),
                    "color": SEASON_COLORS.get(season, [90, 145, 175, 55]),
                    "width": 0.8,
                    "progress_label": "Recorded fixes; habitat-crossing gaps are not connected",
                    "emphasis": "Observed telemetry",
                }
            )

    weights = changed.journeys.set_index("journey_id").scenario_weight
    changed_from_typical = not np.allclose(weights.to_numpy(float), 1.0)
    activated = population_scenario.activated_effects or {}
    # `activated` reflects whether the outcome model *supports* a season's effects, not
    # whether the classroom scenario actually differs from typical conditions -- a
    # typical scenario still reports its (zero-delta) effects here. Without gating on
    # `scenario_selected`, a picked-typical scenario would draw spurious scenario-colored
    # routes/markers built from near-zero, routing-noise-sized deltas.
    scenario_active = bool(activated) and bool(scenario_selected)
    modeled_seasons = set(activated)
    baseline_route_records = []
    scenario_route_records = []
    baseline_routes = {}
    baseline_source_routes = {}
    baseline_raw_routes = {}
    scenario_routes = {}
    scenario_raw_routes = {}
    for season in sorted(baseline_paths.season.dropna().astype(str).unique()):
        baseline_season = baseline_paths[baseline_paths.season.eq(season)].copy()
        baseline_source_route = build_projected_route(baseline_season)
        baseline_source_routes[season] = baseline_source_route
        baseline_raw_route = baseline_source_route
        baseline_raw_route = _extend_route_to_known_terminals(
            baseline_raw_route, season, baseline_season
        )
        baseline_raw_routes[season] = baseline_raw_route
        baseline_route = baseline_raw_route
        if is_marine:
            baseline_route = route_over_water(baseline_route)
        elif is_vulture:
            baseline_route = route_over_land(baseline_route)
        baseline_route["season"] = season
        baseline_routes[season] = baseline_route
        season_color = SEASON_COLORS.get(season, [90, 145, 175, 70])[:3]
        baseline_route_records.append(
            {
                "name": "Baseline population route",
                "journey": "Population projection",
                "season": season.title(),
                "year": "Multiple years",
                "path": baseline_route[["longitude", "latitude"]].values.tolist(),
                "color": season_color + [235],
                "width": 2.6,
                "progress_label": "Full modeled route",
                "emphasis": "Generated center of recorded journeys",
            }
        )

    if scenario_active:
        for season in modeled_seasons:
            if season not in baseline_raw_routes:
                continue
            scenario_season = changed_paths[changed_paths.season.eq(season)].copy()
            if scenario_season.empty:
                continue
            modeled_route = build_projected_route(scenario_season)
            if is_marine:
                modeled_route = _move_deformation_to_display_corridor(
                    modeled_route,
                    baseline_source_routes[season],
                    baseline_raw_routes[season],
                )
            scenario_raw_routes[season] = _anchor_route_endpoints(
                modeled_route,
                baseline_raw_routes[season],
                max_shift_degrees=1.25 if is_marine else None,
            )
        distinct_donors = sorted(
            scenario_raw_routes,
            key=lambda season: _mean_route_shift_km(
                scenario_raw_routes[season], baseline_raw_routes[season]
            ),
            reverse=True,
        )
        for season, route in list(scenario_raw_routes.items()):
            if _mean_route_shift_km(route, baseline_raw_routes[season]) >= 1.0:
                continue
            donor = next(
                (
                    candidate
                    for candidate in distinct_donors
                    if candidate != season
                    and _mean_route_shift_km(
                        scenario_raw_routes[candidate], baseline_raw_routes[candidate]
                    ) >= 1.0
                ),
                None,
            )
            if donor is not None:
                scenario_raw_routes[season] = _transfer_route_deformation(
                    baseline_raw_routes[season],
                    baseline_raw_routes[donor],
                    scenario_raw_routes[donor],
                )

    for season, baseline_route in baseline_routes.items():
        if season in scenario_raw_routes:
            scenario_route = scenario_raw_routes[season]
            if is_marine:
                scenario_route = route_over_water(scenario_route)
            elif is_vulture:
                scenario_route = route_over_land(scenario_route)
            scenario_route["season"] = season
            scenario_routes[season] = scenario_route
            scenario_color = SCENARIO_COLORS.get(season, [164, 92, 255, 255])
            scenario_route_records.append(
                {
                    "name": "Scenario population route",
                    "journey": "Population projection",
                    "season": season.title(),
                    "year": "Multiple years",
                    "path": scenario_route[["longitude", "latitude"]].values.tolist(),
                    "color": scenario_color[:-1] + [245],
                    "width": 4.0,
                    "progress_label": "Known start and end; evidence-weighted route between them",
                    "emphasis": "Generated from seasonal evidence and anchored to recorded endpoints",
                }
            )
    # A scenario marker is only drawn for seasons with an actual distinct scenario
    # route (`scenario_routes`). Seasons where the scenario was "active" but produced
    # a negligible route change have no route of their own to walk, so surfacing a
    # scenario-colored traveler there would show scenario color on what is really the
    # baseline path -- draw nothing extra instead of a misleading duplicate marker.

    baseline_tracks = {
        season: build_animation_track(route) for season, route in baseline_routes.items()
    }
    scenario_tracks = {
        season: build_animation_track(scenario_routes[season]) for season in scenario_routes
    }

    points = baseline_paths[["latitude", "longitude"]].dropna()
    latitude = float(points.latitude.mean())
    angles = points.longitude.map(math.radians)
    longitude = math.degrees(math.atan2(angles.map(math.sin).mean(), angles.map(math.cos).mean()))
    longitude_delta = ((points.longitude - longitude + 180) % 360) - 180
    geographic_span = max(
        float(points.latitude.max() - points.latitude.min()),
        float(longitude_delta.max() - longitude_delta.min()),
        0.01,
    )
    zoom = float(max(0.5, min(9, 7.5 - math.log2(geographic_span))))
    return {
        "observed_records": observed_records,
        "baseline_route_records": baseline_route_records,
        "scenario_route_records": scenario_route_records,
        "baseline_tracks": baseline_tracks,
        "scenario_tracks": scenario_tracks,
        "latitude": latitude,
        "longitude": longitude,
        "zoom": zoom,
    }


def population_route_geometry(population_scenario, focus_seasons=None, scenario_selected=True):
    """Public entry point for precomputing/caching a deck's progress-independent geometry.

    Callers that want geometry to survive across a full Streamlit script rerun (not just
    across fragment ticks within one run -- `_route_geometry`'s identity-keyed cache can't
    do that, since `st.cache_data` returns a fresh copy of `population_scenario` on every
    rerun) should wrap this in their own `st.cache_data`, keyed on `population_scenario`,
    and pass the result to `render_population_deck`.
    """
    return _route_geometry(population_scenario, focus_seasons, scenario_selected)


def render_population_deck(geometry, progress: float = 0.5, playback_mode: bool = False):
    """Assemble a pydeck Deck from geometry built by `population_route_geometry`.

    This only does cheap per-frame work (index into a precomputed animation track,
    assemble pydeck layers), so it's safe to call on every playback tick.
    """
    import pydeck as pdk

    target_progress = float(np.clip(progress, 0.0, 1.0))

    baseline_markers = []
    scenario_markers = []
    for season, track in geometry["baseline_tracks"].items():
        season_color = SEASON_COLORS.get(season, [90, 145, 175, 70])[:3] + [255]
        season_name = season.replace(" migration", "").title()
        baseline_markers.append(
            _route_marker(
                track,
                target_progress,
                name=f"{season_name} baseline traveler",
                color=season_color,
                evidence="Interpolated along the baseline population route",
            )
        )
        scenario_track = geometry["scenario_tracks"].get(season)
        if scenario_track is not None:
            scenario_markers.append(
                _route_marker(
                    scenario_track,
                    target_progress,
                    name=f"{season_name} scenario traveler",
                    color=SCENARIO_COLORS.get(season, [164, 92, 255, 255]),
                    evidence="Interpolated along the constrained scenario route",
                )
            )

    layers = []
    if not playback_mode and geometry["observed_records"]:
        layers.append(pdk.Layer(
            "PathLayer", geometry["observed_records"], get_path="path", get_color="color",
            get_width="width", width_units="pixels", width_min_pixels=0.7,
            width_max_pixels=1.2, pickable=False,
        ))
    layers.append(
        pdk.Layer(
            "PathLayer", geometry["baseline_route_records"], get_path="path", get_color="color",
            get_width="width", width_units="pixels", width_min_pixels=2.2,
            width_max_pixels=3.2, pickable=False,
        )
    )
    if geometry["scenario_route_records"]:
        layers.append(
            pdk.Layer(
                "PathLayer", geometry["scenario_route_records"], get_path="path", get_color="color",
                get_width="width", width_units="pixels", width_min_pixels=3.5,
                width_max_pixels=5.0, pickable=False,
            )
        )
    if baseline_markers:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer", baseline_markers, get_position="position",
                get_radius=30_000, radius_min_pixels=8, radius_max_pixels=12,
                filled=False, stroked=True, get_line_color="color",
                line_width_min_pixels=4, pickable=False,
                transitions={"get_position": 180},
            )
        )
    if scenario_markers:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer", scenario_markers, get_position="position",
                get_fill_color="color", get_radius=18_000, radius_min_pixels=5,
                radius_max_pixels=8, stroked=True, get_line_color=[255, 255, 255, 235],
                line_width_min_pixels=1.5, pickable=False,
                transitions={"get_position": 180},
            )
        )
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=geometry["latitude"], longitude=geometry["longitude"], zoom=geometry["zoom"]
        ),
        tooltip=None,
    )


def build_population_deck(
    population_scenario,
    progress: float = 0.5,
    focus_seasons=None,
    playback_mode: bool = False,
    scenario_selected: bool = True,
):
    """Convenience wrapper: build geometry (cached by `population_scenario` identity
    within this process) and render it in one call. See `population_route_geometry`
    for how to make geometry survive across Streamlit reruns too.
    """
    geometry = population_route_geometry(population_scenario, focus_seasons, scenario_selected)
    return render_population_deck(geometry, progress=progress, playback_mode=playback_mode)
