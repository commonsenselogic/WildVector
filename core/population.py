from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math

import numpy as np
import pandas as pd

from .corridors import assign_corridor_choices
from .forecaster import haversine
from .scenario import WeatherScenario


@dataclass(frozen=True)
class PopulationCorridor:
    corridor: pd.DataFrame
    journeys: pd.DataFrame
    paths: pd.DataFrame
    representative_journey: str
    animals: int
    years: int


@dataclass(frozen=True)
class PopulationScenario:
    baseline: PopulationCorridor
    changed: PopulationCorridor
    endpoint_shift_km: float
    mean_corridor_shift_km: float
    spread_change_km: float
    timing_shift_days: float
    timing_shift_by_season: dict[str, float]
    activated_effects: dict
    explanation: tuple[str, ...]


def _journey_key(frame: pd.DataFrame) -> pd.Series:
    key = frame.animal_id.astype(str) + "|" + frame.year.astype(str)
    if "season" in frame:
        key += "|" + frame.season.astype(str)
    return key


def _unwrap_longitude(values: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(values.astype(float))))


def _wrap_longitude(values) -> np.ndarray:
    return ((np.asarray(values, dtype=float) + 180) % 360) - 180


def _weighted_quantile(values, weights, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return float(np.quantile(values, quantile))
    return float(np.interp(quantile * cumulative[-1], cumulative, values))


def _resample_path(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    ordered = frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
    if len(ordered) < 2:
        return pd.DataFrame()
    elapsed = (ordered.timestamp_utc - ordered.timestamp_utc.iloc[0]).dt.total_seconds().to_numpy(float)
    if elapsed[-1] <= 0:
        return pd.DataFrame()
    source_progress = elapsed / elapsed[-1]
    target = np.linspace(0, 1, bins)
    longitude = _unwrap_longitude(ordered.longitude.to_numpy(float))
    return pd.DataFrame(
        {
            "progress": target,
            "latitude": np.interp(target, source_progress, ordered.latitude),
            "longitude": _wrap_longitude(np.interp(target, source_progress, longitude)),
            "timestamp_utc": pd.to_datetime(
                ordered.timestamp_utc.iloc[0].value
                + target * (ordered.timestamp_utc.iloc[-1].value - ordered.timestamp_utc.iloc[0].value),
                utc=True,
            ),
        }
    )


def _corridor_from_paths(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for progress, group in paths.groupby("progress", sort=True):
        weights = group.get("scenario_weight", pd.Series(1.0, index=group.index)).to_numpy(float)
        latitude = _weighted_quantile(group.latitude, weights, 0.5)
        angles = np.radians(group.longitude.to_numpy(float))
        longitude = math.degrees(
            math.atan2(np.average(np.sin(angles), weights=weights), np.average(np.cos(angles), weights=weights))
        )
        distances = haversine(
            pd.Series(latitude, index=group.index),
            pd.Series(longitude, index=group.index),
            group.latitude,
            group.longitude,
        )
        rows.append(
            {
                "progress": float(progress),
                "latitude": latitude,
                "longitude": longitude,
                "spread_median_km": _weighted_quantile(distances, weights, 0.5),
                "spread_p90_km": _weighted_quantile(distances, weights, 0.9),
                "journeys": int(group.journey_id.nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_projected_route(paths: pd.DataFrame) -> pd.DataFrame:
    """Build a synthetic population route from evidence-weighted journey points.

    Callers must pass one biological season at a time. The returned coordinates
    are modeled progress-wise centers, not observations from an individual.
    """
    if paths.empty:
        raise ValueError("A projected route needs at least one journey path.")
    if "season" in paths and paths.season.nunique() > 1:
        raise ValueError("A projected route cannot combine biological seasons.")
    return _corridor_from_paths(paths)


def _bearing(start_latitude, start_longitude, end_latitude, end_longitude) -> float:
    lat1, lat2 = math.radians(start_latitude), math.radians(end_latitude)
    delta = math.radians(end_longitude - start_longitude)
    y = math.sin(delta) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta)
    return float((math.degrees(math.atan2(y, x)) + 360) % 360)


def _stopover_count(ordered: pd.DataFrame, movement_type: str) -> int:
    elapsed_days = ordered.timestamp_utc.diff().dt.total_seconds() / 86400
    distance = haversine(
        ordered.latitude.shift(), ordered.longitude.shift(), ordered.latitude, ordered.longitude
    )
    pace = distance / elapsed_days.where(elapsed_days > 0)
    threshold = {"aerial": 30.0, "marine": 15.0, "terrestrial": 3.0}.get(movement_type, 10.0)
    slow = elapsed_days.ge(0.5) & pace.lt(threshold)
    return int((slow & ~slow.shift(fill_value=False)).sum())


def build_population_corridor(
    telemetry: pd.DataFrame,
    bins: int = 36,
    minimum_points: int = 4,
) -> PopulationCorridor:
    required = {"animal_id", "year", "timestamp_utc", "latitude", "longitude"}
    if not required.issubset(telemetry):
        raise ValueError(f"Missing population columns: {sorted(required - set(telemetry))}")
    frame = telemetry.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    frame["journey_id"] = _journey_key(frame)
    path_parts, metrics = [], []
    for journey_id, group in frame.groupby("journey_id", sort=True):
        if len(group) < minimum_points:
            continue
        path = _resample_path(group, bins)
        if path.empty:
            continue
        ordered = group.sort_values("timestamp_utc")
        season = str(group.season.iloc[0]) if "season" in group else "migration"
        path["journey_id"] = journey_id
        path["animal_id"] = str(group.animal_id.iloc[0])
        path["year"] = int(group.year.iloc[0])
        path["season"] = season
        path["movement_type"] = (
            str(group.movement_type.iloc[0]) if "movement_type" in group else "aerial"
        )
        for identity_column in ("species", "population"):
            if identity_column in group:
                path[identity_column] = str(group[identity_column].iloc[0])
        path["scenario_weight"] = 1.0
        path_parts.append(path)
        distance = haversine(
            ordered.latitude.shift(), ordered.longitude.shift(), ordered.latitude, ordered.longitude
        ).fillna(0).sum()
        net_distance = float(
            haversine(
                pd.Series([ordered.latitude.iloc[0]]),
                pd.Series([ordered.longitude.iloc[0]]),
                pd.Series([ordered.latitude.iloc[-1]]),
                pd.Series([ordered.longitude.iloc[-1]]),
            ).iloc[0]
        )
        metrics.append(
            {
                "journey_id": journey_id,
                "animal_id": str(group.animal_id.iloc[0]),
                "year": int(group.year.iloc[0]),
                "season": season,
                "points": len(group),
                "distance_km": float(distance),
                "net_distance_km": net_distance,
                "duration_days": float((ordered.timestamp_utc.iloc[-1] - ordered.timestamp_utc.iloc[0]).total_seconds() / 86400),
                "start_day": float(ordered.timestamp_utc.iloc[0].dayofyear),
                "end_day": float(ordered.timestamp_utc.iloc[-1].dayofyear),
                "pace_km_day": float(distance) / max(
                    (ordered.timestamp_utc.iloc[-1] - ordered.timestamp_utc.iloc[0]).total_seconds() / 86400,
                    0.25,
                ),
                "stopover_count": _stopover_count(
                    ordered,
                    str(group.movement_type.iloc[0]) if "movement_type" in group else "aerial",
                ),
                "route_bearing": _bearing(
                    ordered.latitude.iloc[0], ordered.longitude.iloc[0],
                    ordered.latitude.iloc[-1], ordered.longitude.iloc[-1],
                ),
                "route_25_latitude": float(path.iloc[round((len(path) - 1) * 0.25)].latitude),
                "route_25_longitude": float(path.iloc[round((len(path) - 1) * 0.25)].longitude),
                "route_50_latitude": float(path.iloc[round((len(path) - 1) * 0.50)].latitude),
                "route_50_longitude": float(path.iloc[round((len(path) - 1) * 0.50)].longitude),
                "route_75_latitude": float(path.iloc[round((len(path) - 1) * 0.75)].latitude),
                "route_75_longitude": float(path.iloc[round((len(path) - 1) * 0.75)].longitude),
            }
        )
    if not path_parts:
        raise ValueError("No seasonal journey has enough points for a population view.")
    paths = pd.concat(path_parts, ignore_index=True)
    corridor = _corridor_from_paths(paths)
    deviation = paths.merge(
        corridor[["progress", "latitude", "longitude"]],
        on="progress", suffixes=("", "_corridor"),
    )
    deviation["corridor_distance_km"] = haversine(
        deviation.latitude, deviation.longitude,
        deviation.latitude_corridor, deviation.longitude_corridor,
    )
    scores = deviation.groupby("journey_id").corridor_distance_km.mean().sort_values()
    representative = str(scores.index[0])
    journeys = pd.DataFrame(metrics)
    journeys["corridor_choice"] = assign_corridor_choices(journeys)
    journeys["corridor_error_km"] = journeys.journey_id.map(scores)
    journeys["representative"] = journeys.journey_id.eq(representative)
    journeys["scenario_weight"] = 1.0
    paths["representative"] = paths.journey_id.eq(representative)
    return PopulationCorridor(
        corridor=corridor,
        journeys=journeys.sort_values("corridor_error_km").reset_index(drop=True),
        paths=paths,
        representative_journey=representative,
        animals=int(frame.animal_id.nunique()),
        years=int(frame.year.nunique()),
    )


def _standardize_by_season(frame: pd.DataFrame, column: str) -> pd.Series:
    def standardize(values: pd.Series) -> pd.Series:
        spread = float(values.std(ddof=0))
        return (values - values.mean()) / spread if spread > 0 else values * 0
    return frame.groupby("season", group_keys=False)[column].apply(standardize)


def _alignment(bearing: pd.Series, direction: float | None) -> pd.Series:
    if direction is None:
        return pd.Series(1.0, index=bearing.index)
    return np.cos(np.radians(bearing.astype(float) - float(direction)))


def _paired_season_effects(
    journeys: pd.DataFrame,
    activated_effects: dict[str, dict[str, dict]] | None,
) -> dict[str, dict[str, dict]]:
    """Extend a validated seasonal response across the paired migration cycle.

    The source season keeps its forward-tested model result. The other season
    receives the same standardized continuous response, or the same probability-
    point corridor response, centered on that season's recorded journeys. This is
    deliberately labeled as a paired-season projection rather than independent
    validation for the receiving season.
    """
    if not activated_effects:
        return {}
    available_seasons = [
        str(season) for season in journeys.season.dropna().astype(str).unique()
    ]
    expanded = deepcopy(activated_effects)
    for season, effects in expanded.items():
        for effect in effects.values():
            effect.setdefault("support", "validated")
            effect.setdefault("evidence_season", season)

    outcome_columns = {
        "departure_date": "start_day",
        "arrival_date": "end_day",
        "migration_duration": "duration_days",
        "migration_pace": "pace_km_day",
        "stopovers": "stopover_count",
    }
    donors = [(season, effects) for season, effects in expanded.items() if effects]
    if not donors:
        return expanded
    for target_season in available_seasons:
        if expanded.get(target_season):
            continue
        donor_season, donor_effects = donors[0]
        target = journeys[journeys.season.eq(target_season)]
        donor = journeys[journeys.season.eq(donor_season)]
        if target.empty:
            continue
        projected: dict[str, dict] = {}
        for outcome, donor_effect in donor_effects.items():
            effect = deepcopy(donor_effect)
            effect["support"] = "paired-season projection"
            effect["evidence_season"] = donor_season
            if donor_effect.get("kind") == "continuous":
                column = outcome_columns.get(outcome)
                if column is None or column not in target:
                    continue
                target_baseline = float(target[column].mean())
                donor_spread = float(donor[column].std(ddof=0)) if column in donor else 0.0
                target_spread = float(target[column].std(ddof=0))
                donor_delta = float(donor_effect.get("delta", 0.0))
                target_delta = (
                    donor_delta * target_spread / donor_spread
                    if donor_spread > 0 and target_spread > 0
                    else donor_delta
                )
                target_delta = float(np.clip(target_delta, -1.5 * max(target_spread, abs(donor_delta), 1e-6), 1.5 * max(target_spread, abs(donor_delta), 1e-6)))
                target_scenario = target_baseline + target_delta
                if outcome in {"migration_duration", "migration_pace", "stopovers"}:
                    target_scenario = max(0.0, target_scenario)
                    target_delta = target_scenario - target_baseline
                effect.update(
                    baseline=target_baseline,
                    scenario=target_scenario,
                    delta=target_delta,
                )
            elif donor_effect.get("kind") == "classification":
                target_counts = target.corridor_choice.astype(str).value_counts(normalize=True)
                labels = sorted(
                    set(target_counts.index)
                    | set(donor_effect.get("baseline", {}))
                    | set(donor_effect.get("scenario", {}))
                )
                baseline_prob = {
                    label: float(target_counts.get(label, 0.0)) for label in labels
                }
                shifted = {
                    label: max(
                        0.0,
                        baseline_prob[label]
                        + float(donor_effect.get("scenario", {}).get(label, 0.0))
                        - float(donor_effect.get("baseline", {}).get(label, 0.0)),
                    )
                    for label in labels
                }
                total = sum(shifted.values())
                if total <= 0:
                    continue
                effect.update(
                    baseline=baseline_prob,
                    scenario={label: value / total for label, value in shifted.items()},
                )
            else:
                continue
            projected[outcome] = effect
        if projected:
            expanded[target_season] = projected
    return expanded


def simulate_population_scenario(
    telemetry: pd.DataFrame,
    scenario: WeatherScenario,
    movement_type: str,
    bins: int = 36,
    activated_effects: dict[str, dict[str, dict]] | None = None,
) -> PopulationScenario:
    """Reweight recordings and derive an evidence-based scenario corridor."""
    baseline = build_population_corridor(telemetry, bins=bins)
    journeys = baseline.journeys.copy()
    if activated_effects is not None:
        activated_effects = _paired_season_effects(journeys, activated_effects)
    score = pd.Series(0.0, index=journeys.index)
    if activated_effects is None:
        # Backward-compatible classroom heuristic. The application passes an explicit
        # activation dictionary and therefore never enters this branch.
        pace_z = _standardize_by_season(journeys, "pace_km_day")
        wind_coupling = {"aerial": 0.9, "marine": 0.35, "terrestrial": 0.2}.get(movement_type, 0.25)
        wind_pattern = pace_z if scenario.wind_direction_deg is None else _alignment(
            journeys.route_bearing, scenario.wind_direction_deg
        )
        score += scenario.wind_speed_change_kmh / 20.0 * wind_pattern * wind_coupling
        if movement_type == "marine":
            current_pattern = pace_z if scenario.current_direction_deg is None else _alignment(
                journeys.route_bearing, scenario.current_direction_deg
            )
            score += scenario.current_speed_change_kmh / 2.0 * current_pattern
        timing_z = _standardize_by_season(journeys, "start_day")
        seasonal_sign = journeys.season.map(
            {"spring migration": -1.0, "fall migration": 1.0}
        ).fillna(0.0)
        score += scenario.temperature_change_c / 8.0 * timing_z * seasonal_sign * 0.75
        score += scenario.pressure_trend_hpa_per_day / 3.0 * timing_z * 0.15
    else:
        outcome_columns = {
            "departure_date": "start_day",
            "arrival_date": "end_day",
            "migration_duration": "duration_days",
            "migration_pace": "pace_km_day",
            "stopovers": "stopover_count",
        }
        for season, effects in activated_effects.items():
            season_mask = journeys.season.eq(season)
            for outcome, effect in effects.items():
                if outcome == "corridor_choice" and effect.get("kind") == "classification":
                    baseline_prob = effect.get("baseline", {})
                    scenario_prob = effect.get("scenario", {})
                    for corridor, probability in scenario_prob.items():
                        reference = max(float(baseline_prob.get(corridor, 0.0)), 1e-6)
                        ratio = max(float(probability), 1e-6) / reference
                        mask = season_mask & journeys.corridor_choice.eq(str(corridor))
                        score.loc[mask] += float(np.clip(np.log(ratio), -1.5, 1.5))
                    continue
                column = outcome_columns.get(outcome)
                if column is None or effect.get("kind") != "continuous" or not season_mask.any():
                    continue
                spread = float(journeys.loc[season_mask, column].std(ddof=0))
                if spread <= 0:
                    continue
                centered = (
                    journeys.loc[season_mask, column] - journeys.loc[season_mask, column].mean()
                ) / spread
                score.loc[season_mask] += centered * float(
                    np.clip(effect.get("delta", 0.0) / spread, -1.5, 1.5)
                )

    journeys["scenario_score"] = score.clip(-2.0, 2.0)
    journeys["scenario_weight"] = journeys.groupby("season")["scenario_score"].transform(
        lambda values: np.exp(values) / np.exp(values).mean()
    ).clip(0.35, 2.5)
    if scenario.is_typical:
        journeys["scenario_weight"] = 1.0

    paths = baseline.paths.copy()
    weight_map = journeys.set_index("journey_id").scenario_weight
    paths["scenario_weight"] = paths.journey_id.map(weight_map).fillna(1.0)
    changed = replace(
        baseline,
        paths=paths,
        journeys=journeys,
        corridor=_corridor_from_paths(paths),
    )
    paired = baseline.corridor.merge(changed.corridor, on="progress", suffixes=("_baseline", "_scenario"))
    shifts = haversine(
        paired.latitude_baseline, paired.longitude_baseline,
        paired.latitude_scenario, paired.longitude_scenario,
    )
    spread_change = float(paired.spread_p90_km_scenario.mean() - paired.spread_p90_km_baseline.mean())
    timing_by_season = {}
    for season, group in journeys.groupby("season"):
        timing_by_season[str(season)] = float(
            np.average(group.start_day, weights=group.scenario_weight) - group.start_day.mean()
        )
    timing_shift = float(np.mean(list(timing_by_season.values())))
    timing_text = ", ".join(
        f"{season.replace(' migration', '')} {shift:+.1f} days"
        for season, shift in timing_by_season.items()
    )
    active_names = sorted(
        {outcome for effects in (activated_effects or {}).values() for outcome in effects}
    )
    if activated_effects is not None and not active_names:
        explanation = (
            "No historical weather effect passed both baselines for these seasons.",
            "The controls therefore leave every recorded journey unchanged.",
        )
    else:
        activation_text = (
            "Validated outcomes used: " + ", ".join(name.replace("_", " ") for name in active_names) + "."
            if activated_effects is not None
            else "This is the legacy classroom comparison, not an activated historical effect."
        )
        explanation = (
            "Recorded journeys remain unchanged; the scenario corridor is a modeled population route generated from their evidence-weighted coordinates.",
            activation_text,
            f"Emphasized departure timing by season: {timing_text}.",
        )
    return PopulationScenario(
        baseline=baseline,
        changed=changed,
        endpoint_shift_km=float(shifts.iloc[-1]),
        mean_corridor_shift_km=float(shifts.mean()),
        spread_change_km=spread_change,
        timing_shift_days=timing_shift,
        timing_shift_by_season=timing_by_season,
        activated_effects=activated_effects or {},
        explanation=explanation,
    )
