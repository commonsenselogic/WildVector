from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .catalog import DEFAULT_CATALOG_ROOT, slug
from .environmental_history import build_sampling_points
from .forecaster import haversine
from .population import PopulationScenario, build_population_corridor
from .scenario import WeatherScenario, _advance


BASELINE_FEATURES = ["progress_sin", "progress_cos", "year_day_sin", "year_day_cos"]
ENVIRONMENT_FEATURES = [
    "temperature_2m",
    "surface_pressure",
    "precipitation",
    "wind_u_component_10m",
    "wind_v_component_10m",
    "sea_surface_temperature",
    "marine_current_u",
    "marine_current_v",
]


@dataclass(frozen=True)
class ModelValidation:
    species: str
    population: str
    season: str
    years: int
    journeys: int
    samples: int
    folds: int
    baseline_rmse_km_day: float
    environmental_rmse_km_day: float
    skill_vs_seasonal_baseline: float
    status: str
    feature_columns: tuple[str, ...]


def _segment_components(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["animal_id", "year", "progress_bin"]).copy()
    group = out.groupby(["animal_id", "year"], sort=False)
    mean_latitude = np.radians((out.latitude + group.latitude.shift()) / 2)
    out["target_north_km_day"] = (out.latitude - group.latitude.shift()) * 110.574
    out["target_east_km_day"] = (
        (out.longitude - group.longitude.shift()) * 111.320 * np.cos(mean_latitude)
    )
    elapsed = (pd.to_datetime(out.date) - pd.to_datetime(group.date.shift())).dt.days
    elapsed = elapsed.where(elapsed > 0)
    out["target_north_km_day"] /= elapsed
    out["target_east_km_day"] /= elapsed
    progress = out.progress_bin / group.progress_bin.transform("max").replace(0, 1)
    out["progress_sin"] = np.sin(progress * 2 * np.pi)
    out["progress_cos"] = np.cos(progress * 2 * np.pi)
    day = pd.to_datetime(out.date).dt.dayofyear
    out["year_day_sin"] = np.sin(day / 365.25 * 2 * np.pi)
    out["year_day_cos"] = np.cos(day / 365.25 * 2 * np.pi)
    return out.replace([np.inf, -np.inf], np.nan)


def training_frame(
    telemetry: pd.DataFrame,
    environment: pd.DataFrame,
    bins: int = 12,
) -> pd.DataFrame:
    points = build_sampling_points(telemetry, bins=bins)
    keys = [
        "species",
        "population",
        "movement_type",
        "season",
        "year",
        "animal_id",
        "progress_bin",
        "date",
    ]
    environmental_columns = [column for column in ENVIRONMENT_FEATURES if column in environment]
    merged = points.merge(environment[keys + environmental_columns], on=keys, how="left")
    return _segment_components(merged)


def _usable_environment_features(frame: pd.DataFrame) -> list[str]:
    usable = []
    for column in ENVIRONMENT_FEATURES:
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().mean() >= 0.2 and numeric.nunique(dropna=True) > 1:
            frame[column] = numeric.fillna(numeric.median())
            usable.append(column)
    return usable


def fit_population_response_model(
    telemetry: pd.DataFrame,
    environment: pd.DataFrame,
    bins: int = 12,
):
    if telemetry.empty:
        raise ValueError("Population telemetry is empty.")
    identity = telemetry.iloc[0]
    frame = training_frame(telemetry, environment, bins=bins)
    targets = ["target_east_km_day", "target_north_km_day"]
    frame = frame.dropna(subset=targets).copy()
    years = sorted(int(value) for value in frame.year.unique())
    if len(years) < 3:
        raise ValueError("At least three migration years are required for held-out-year validation.")
    environment_features = _usable_environment_features(frame)
    if not environment_features:
        raise ValueError("No multi-year environmental features have enough coverage to train.")
    full_features = BASELINE_FEATURES + environment_features
    baseline_template = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    full_template = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    splitter = LeaveOneGroupOut()
    baseline_errors = []
    environmental_errors = []
    fold_count = 0
    x_baseline = frame[BASELINE_FEATURES]
    x_full = frame[full_features]
    y = frame[targets]
    groups = frame.year
    for train, test in splitter.split(x_full, y, groups):
        if len(train) < 12 or len(test) < 2:
            continue
        baseline = clone(baseline_template).fit(x_baseline.iloc[train], y.iloc[train])
        full = clone(full_template).fit(x_full.iloc[train], y.iloc[train])
        baseline_errors.append(
            mean_squared_error(y.iloc[test], baseline.predict(x_baseline.iloc[test])) ** 0.5
        )
        environmental_errors.append(
            mean_squared_error(y.iloc[test], full.predict(x_full.iloc[test])) ** 0.5
        )
        fold_count += 1
    if not fold_count:
        raise ValueError("No held-out migration year had enough samples for validation.")
    baseline_rmse = float(np.mean(baseline_errors))
    environmental_rmse = float(np.mean(environmental_errors))
    skill = float(1 - environmental_rmse / baseline_rmse) if baseline_rmse > 0 else 0.0
    status = (
        "validated for classroom scenarios"
        if fold_count >= 3 and skill > 0.05
        else "environment model did not beat the seasonal baseline"
    )
    final_model = full_template.fit(x_full, y)
    validation = ModelValidation(
        species=str(identity.species),
        population=str(identity.population),
        season=str(identity.season),
        years=len(years),
        journeys=int(frame[["animal_id", "year"]].drop_duplicates().shape[0]),
        samples=len(frame),
        folds=fold_count,
        baseline_rmse_km_day=baseline_rmse,
        environmental_rmse_km_day=environmental_rmse,
        skill_vs_seasonal_baseline=skill,
        status=status,
        feature_columns=tuple(full_features),
    )
    bundle = {
        "model": final_model,
        "features": full_features,
        "fill_values": {
            column: float(pd.to_numeric(frame[column], errors="coerce").median())
            for column in environment_features
        },
        "route_bins": bins,
        "targets": targets,
        "validation": asdict(validation),
    }
    return bundle, validation


def _changed_environment(frame: pd.DataFrame, scenario: WeatherScenario) -> pd.DataFrame:
    changed = frame.copy()
    if "temperature_2m" in changed:
        changed["temperature_2m"] += scenario.temperature_change_c
    if "surface_pressure" in changed:
        progress = changed.progress_bin / changed.groupby("year").progress_bin.transform("max").replace(0, 1)
        changed["surface_pressure"] += scenario.pressure_trend_hpa_per_day * progress * 3

    def adjust_vectors(u_column, v_column, speed_change, direction):
        if u_column not in changed or v_column not in changed:
            return
        observed_u = pd.to_numeric(changed[u_column], errors="coerce").fillna(0)
        observed_v = pd.to_numeric(changed[v_column], errors="coerce").fillna(0)
        observed_speed = np.hypot(observed_u, observed_v)
        if direction is None:
            angles = np.arctan2(observed_u, observed_v)
        else:
            angles = np.full(len(changed), np.radians(float(direction)))
        target_speed = np.maximum(0, observed_speed + float(speed_change))
        changed[u_column] = target_speed * np.sin(angles)
        changed[v_column] = target_speed * np.cos(angles)

    adjust_vectors(
        "wind_u_component_10m",
        "wind_v_component_10m",
        scenario.wind_speed_change_kmh,
        scenario.wind_direction_deg,
    )
    adjust_vectors(
        "marine_current_u",
        "marine_current_v",
        scenario.current_speed_change_kmh,
        scenario.current_direction_deg,
    )
    return changed


def apply_population_response_model(
    telemetry: pd.DataFrame,
    environment: pd.DataFrame,
    bundle: dict,
    scenario: WeatherScenario,
    bins: int = 36,
) -> PopulationScenario:
    """Disabled: velocity deltas cannot safely generate unconstrained map coordinates."""
    raise RuntimeError(
        "Historical population models are diagnostic only. Use "
        "simulate_population_scenario to reweight recorded journeys without moving coordinates."
    )
    # The former coordinate-displacement implementation remains below temporarily for
    # artifact compatibility, but is unreachable and scheduled for removal.
    validation = bundle.get("validation", {})
    bins = int(bundle.get("route_bins", bins))
    if validation.get("status") != "validated for classroom scenarios":
        raise ValueError("The environmental model has not beaten its seasonal baseline.")
    baseline = build_population_corridor(telemetry, bins=bins)
    if scenario.is_typical:
        return PopulationScenario(
            baseline,
            baseline,
            0.0,
            0.0,
            0.0,
            ("Typical conditions selected; the population corridors match.",),
        )
    points = build_sampling_points(telemetry, bins=bins)
    keys = [
        "species",
        "population",
        "movement_type",
        "season",
        "year",
        "animal_id",
        "progress_bin",
        "date",
    ]
    features = [column for column in ENVIRONMENT_FEATURES if column in environment]
    frame = points.merge(environment[keys + features], on=keys, how="left")
    frame = _segment_components(frame)
    for column, value in bundle.get("fill_values", {}).items():
        if column not in frame:
            frame[column] = value
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(value)
    changed_features = _changed_environment(frame, scenario)
    model_features = bundle["features"]
    baseline_velocity = bundle["model"].predict(frame[model_features])
    changed_velocity = bundle["model"].predict(changed_features[model_features])
    delta_velocity = changed_velocity - baseline_velocity
    offsets = []
    for year, group in frame.assign(
        delta_east=delta_velocity[:, 0], delta_north=delta_velocity[:, 1]
    ).groupby("year", sort=True):
        group = group.sort_values("progress_bin")
        east_offset = north_offset = 0.0
        previous_date = None
        for point in group.itertuples(index=False):
            current_date = pd.Timestamp(point.date)
            elapsed = 0 if previous_date is None else max(0, (current_date - previous_date).days)
            east_offset += float(point.delta_east) * elapsed
            north_offset += float(point.delta_north) * elapsed
            offsets.append(
                {
                    "year": int(year),
                    "progress": float(point.progress_bin / max(group.progress_bin.max(), 1)),
                    "east_offset": east_offset,
                    "north_offset": north_offset,
                }
            )
            previous_date = current_date
    offsets = pd.DataFrame(offsets)
    changed_paths = baseline.paths.merge(offsets, on=["year", "progress"], how="left")
    changed_paths[["east_offset", "north_offset"]] = changed_paths[
        ["east_offset", "north_offset"]
    ].fillna(0)
    coordinates = changed_paths.apply(
        lambda row: _advance(
            row.latitude, row.longitude, row.east_offset, row.north_offset
        ),
        axis=1,
    )
    changed_paths["latitude"] = [value[0] for value in coordinates]
    changed_paths["longitude"] = [value[1] for value in coordinates]
    changed_frame = changed_paths[
        ["animal_id", "year", "timestamp_utc", "latitude", "longitude"]
    ].copy()
    changed = build_population_corridor(changed_frame, bins=bins)
    paired = baseline.corridor.merge(
        changed.corridor, on="progress", suffixes=("_baseline", "_scenario")
    )
    shifts = haversine(
        paired.latitude_baseline,
        paired.longitude_baseline,
        paired.latitude_scenario,
        paired.longitude_scenario,
    )
    endpoint_shift = float(shifts.iloc[-1])
    mean_shift = float(shifts.mean())
    spread_change = float(
        paired.spread_p90_km_scenario.mean() - paired.spread_p90_km_baseline.mean()
    )
    return PopulationScenario(
        baseline,
        changed,
        endpoint_shift,
        mean_shift,
        spread_change,
        (
            f"The held-out-year model shifts the corridor by {mean_shift:,.0f} km on average.",
            f"The tracked-animal spread changes by {spread_change:+,.0f} km.",
            f"Historical skill over the seasonal baseline was {validation['skill_vs_seasonal_baseline']:.1%}.",
        ),
    )


def save_population_model(
    bundle: dict,
    validation: ModelValidation,
    catalog_root: Path | str = DEFAULT_CATALOG_ROOT,
) -> tuple[Path, Path]:
    root = Path(catalog_root) / "models"
    key = "__".join(
        [slug(validation.species), slug(validation.population), slug(validation.season)]
    )
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / f"{key}.joblib"
    metrics_path = root / f"{key}.json"
    temporary_model = root / f".{key}.joblib.tmp"
    temporary_metrics = root / f".{key}.json.tmp"
    joblib.dump(bundle, temporary_model)
    temporary_metrics.write_text(
        json.dumps(asdict(validation), indent=2), encoding="utf-8"
    )
    temporary_model.replace(model_path)
    temporary_metrics.replace(metrics_path)
    return model_path, metrics_path
