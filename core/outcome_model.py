from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .catalog import DEFAULT_CATALOG_ROOT, slug
from .corridors import assign_corridor_choices
from .forecaster import haversine
from .population import _unwrap_longitude, _wrap_longitude
from .population_model import ENVIRONMENT_FEATURES as LEGACY_ENVIRONMENT_FEATURES
from .predictor_enrichment import PREDICTOR_FEATURE_GROUPS
from .scenario import WeatherScenario


CONTINUOUS_OUTCOMES = {
    "departure_date": ("departure_day", "days"),
    "arrival_date": ("arrival_day", "days"),
    "migration_duration": ("duration_days", "days"),
    "migration_pace": ("pace_km_day", "km/day"),
    "stopovers": ("stopover_count", "stops"),
}
MINIMUM_TRAIN_YEARS = 3
MINIMUM_ACTIVATION_FOLDS = 3
MINIMUM_SKILL = 0.05
MINIMUM_FOLD_WIN_RATE = 0.60
MINIMUM_EFFECT_STABILITY = 0.67
MINIMUM_SCENARIO_PROBABILITY_CHANGE = 0.005
MINIMUM_FEATURE_COVERAGE = 0.35
OUTCOME_WINDOWS = {
    "departure_date": ("departure_30d__", "departure_7d__"),
    "arrival_date": ("departure_30d__", "departure_7d__", "route_early_7d__", "route_middle_7d__", "route_late_7d__"),
    "migration_duration": ("departure_7d__", "route_early_7d__", "route_middle_7d__", "route_late_7d__"),
    "migration_pace": ("departure_7d__", "route_early_7d__", "route_middle_7d__", "route_late_7d__"),
    "stopovers": ("route_early_7d__", "route_middle_7d__", "route_late_7d__"),
    "corridor_choice": ("departure_30d__", "departure_7d__"),
}
NON_FEATURE_SUFFIXES = (
    "wind_direction_10m_dominant", "route_bearing_degrees", "days_requested", "days_observed",
)


@dataclass(frozen=True)
class OutcomeValidation:
    species: str
    population: str
    season: str
    outcome: str
    outcome_kind: str
    unit: str
    years: int
    folds: int
    seasonal_error: float
    persistence_error: float
    environmental_error: float
    skill_vs_seasonal: float
    skill_vs_persistence: float
    fold_win_rate: float
    effect_stability: float
    environment_coverage: float
    status: str
    feature_columns: tuple[str, ...]
    top_effects: tuple[dict, ...]


def _journey_id(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.animal_id.astype(str)
        + "|"
        + frame.year.astype(str)
        + "|"
        + frame.season.astype(str)
    )


def _route_features(ordered: pd.DataFrame) -> dict[str, float]:
    elapsed = (ordered.timestamp_utc - ordered.timestamp_utc.iloc[0]).dt.total_seconds().to_numpy(float)
    if elapsed[-1] <= 0:
        progress = np.linspace(0, 1, len(ordered))
    else:
        progress = elapsed / elapsed[-1]
    longitude = _unwrap_longitude(ordered.longitude.to_numpy(float))
    features = {}
    for percent, target in ((25, 0.25), (50, 0.50), (75, 0.75)):
        features[f"route_{percent}_latitude"] = float(np.interp(target, progress, ordered.latitude))
        features[f"route_{percent}_longitude"] = float(
            _wrap_longitude([np.interp(target, progress, longitude)])[0]
        )
    return features


def _count_stopovers(ordered: pd.DataFrame, movement_type: str) -> int:
    elapsed_days = ordered.timestamp_utc.diff().dt.total_seconds() / 86400
    distance = haversine(
        ordered.latitude.shift(), ordered.longitude.shift(), ordered.latitude, ordered.longitude
    )
    pace = distance / elapsed_days.where(elapsed_days > 0)
    threshold = {"aerial": 30.0, "marine": 15.0, "terrestrial": 3.0}.get(movement_type, 10.0)
    slow = elapsed_days.ge(0.5) & pace.lt(threshold)
    return int((slow & ~slow.shift(fill_value=False)).sum())


def extract_journey_outcomes(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Calculate observable outcomes for each animal-season-year journey."""
    required = {
        "species", "population", "movement_type", "season", "year", "animal_id",
        "timestamp_utc", "latitude", "longitude",
    }
    if not required.issubset(telemetry):
        raise ValueError(f"Missing outcome columns: {sorted(required - set(telemetry))}")
    frame = telemetry.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    frame["journey_id"] = _journey_id(frame)
    rows = []
    for journey_id, group in frame.groupby("journey_id", sort=True):
        ordered = group.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
        if len(ordered) < 4:
            continue
        duration = float((ordered.timestamp_utc.iloc[-1] - ordered.timestamp_utc.iloc[0]).total_seconds() / 86400)
        if duration <= 0:
            continue
        distance = float(
            haversine(
                ordered.latitude.shift(), ordered.longitude.shift(),
                ordered.latitude, ordered.longitude,
            ).fillna(0).sum()
        )
        identity = ordered.iloc[0]
        rows.append(
            {
                "species": str(identity.species),
                "population": str(identity.population),
                "movement_type": str(identity.movement_type),
                "season": str(identity.season),
                "year": int(identity.year),
                "animal_id": str(identity.animal_id),
                "journey_id": journey_id,
                "departure_day": float(ordered.timestamp_utc.iloc[0].dayofyear),
                "arrival_day": float(ordered.timestamp_utc.iloc[-1].dayofyear),
                "duration_days": duration,
                "distance_km": distance,
                "pace_km_day": distance / duration,
                "stopover_count": _count_stopovers(ordered, str(identity.movement_type)),
                **_route_features(ordered),
            }
        )
    outcomes = pd.DataFrame(rows)
    if outcomes.empty:
        return outcomes
    outcomes["corridor_choice"] = assign_corridor_choices(outcomes)
    return outcomes


def population_year_outcomes(telemetry: pd.DataFrame) -> pd.DataFrame:
    journeys = extract_journey_outcomes(telemetry)
    if journeys.empty:
        return journeys
    group_columns = ["species", "population", "movement_type", "season", "year"]
    continuous = [column for column, _ in CONTINUOUS_OUTCOMES.values()]
    annual = journeys.groupby(group_columns, as_index=False)[continuous].median()
    counts = journeys.groupby(group_columns).size().rename("journeys").reset_index()
    dominant = (
        journeys.groupby(group_columns).corridor_choice
        .agg(lambda values: values.mode().iloc[0])
        .rename("corridor_choice")
        .reset_index()
    )
    return annual.merge(counts, on=group_columns).merge(dominant, on=group_columns)


def outcome_training_frame(telemetry: pd.DataFrame, environment: pd.DataFrame) -> pd.DataFrame:
    if "journey_id" in environment and "window_name" in environment:
        journeys = extract_journey_outcomes(telemetry)
        if journeys.empty or environment.empty:
            return journeys.iloc[0:0]
        keys = [
            "species", "population", "movement_type", "season", "year", "animal_id", "journey_id",
        ]
        excluded = set(keys) | {
            "window_name", "anchor_date", "start_date", "end_date", "latitude", "longitude",
            "environment_source",
        }
        values = [
            column for column in environment.columns
            if column not in excluded
            and not column.endswith(NON_FEATURE_SUFFIXES)
            and pd.api.types.is_numeric_dtype(environment[column])
        ]
        weather = environment.pivot_table(
            index=keys, columns="window_name", values=values, aggfunc="mean"
        )
        weather.columns = [f"{window}__{feature}" for feature, window in weather.columns]
        weather = weather.reset_index()
        return journeys.merge(weather, on=keys, how="inner").sort_values(
            ["year", "journey_id"]
        ).reset_index(drop=True)
    annual = population_year_outcomes(telemetry)
    if annual.empty or environment.empty:
        return annual.iloc[0:0]
    keys = ["species", "population", "movement_type", "season", "year"]
    available = [column for column in LEGACY_ENVIRONMENT_FEATURES if column in environment]
    weather = environment.groupby(keys, as_index=False)[available].mean()
    return annual.merge(weather, on=keys, how="inner").sort_values("year").reset_index(drop=True)


def _usable_features(frame: pd.DataFrame, prefixes: tuple[str, ...] | None = None) -> tuple[list[str], float]:
    usable = []
    coverage_values = []
    candidates = [
        column for column in frame
        if column in LEGACY_ENVIRONMENT_FEATURES or "__" in column
    ]
    if prefixes:
        candidates = [column for column in candidates if column.startswith(prefixes)]
    for column in candidates:
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        coverage = float(numeric.notna().mean())
        if coverage >= MINIMUM_FEATURE_COVERAGE and numeric.nunique(dropna=True) >= 3:
            frame[column] = numeric
            usable.append(column)
            coverage_values.append(coverage)
    return usable, float(np.mean(coverage_values)) if coverage_values else 0.0


def _feature_source(feature: str) -> str:
    raw = feature.split("__", 1)[-1]
    for source, names in PREDICTOR_FEATURE_GROUPS.items():
        if raw in names:
            return source
    return "atmosphere_daily"


def _source_feature_groups(features: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for feature in features:
        groups.setdefault(_feature_source(feature), []).append(feature)
    return groups


def _accepted(validation: OutcomeValidation | None) -> bool:
    return bool(validation and validation.status == "active historical association")


def _top_effects(model, features: list[str], fold_coefficients: list[np.ndarray]) -> tuple[tuple[dict, ...], float]:
    coefficients = np.asarray(model.named_steps["ridge"].coef_, dtype=float)
    ranking = np.argsort(np.abs(coefficients))[::-1][:3]
    effects = []
    stability_values = []
    for index in ranking:
        fold_signs = [np.sign(values[index]) for values in fold_coefficients if len(values) > index]
        final_sign = np.sign(coefficients[index])
        stability = float(np.mean(np.asarray(fold_signs) == final_sign)) if fold_signs else 0.0
        stability_values.append(stability)
        effects.append(
            {
                "feature": features[index],
                "direction": "higher" if coefficients[index] > 0 else "lower",
                "standardized_coefficient": float(coefficients[index]),
                "fold_sign_stability": stability,
            }
        )
    return tuple(effects), max(stability_values, default=0.0)


def _continuous_model(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    identity,
    outcome: str,
    unit: str,
    coverage: float,
):
    model_errors, seasonal_errors, persistence_errors = [], [], []
    fold_coefficients, fold_wins = [], []
    template = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=4.0),
    )
    years = sorted(frame.year.unique())
    for test_year in years[MINIMUM_TRAIN_YEARS:]:
        train, test = frame[frame.year < test_year], frame[frame.year == test_year]
        if train.empty or test.empty:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=4.0),
        )
        model.fit(train[features], train[target])
        actual = test[target].to_numpy(float)
        fold_model_errors = model.predict(test[features]).astype(float) - actual
        fold_seasonal_errors = np.repeat(float(train[target].median()), len(test)) - actual
        previous = train[train.year.eq(train.year.max())][target].median()
        fold_persistence_errors = np.repeat(float(previous), len(test)) - actual
        model_errors.extend(fold_model_errors)
        seasonal_errors.extend(fold_seasonal_errors)
        persistence_errors.extend(fold_persistence_errors)
        fold_wins.append(
            np.sqrt(np.mean(np.square(fold_model_errors)))
            < min(
                np.sqrt(np.mean(np.square(fold_seasonal_errors))),
                np.sqrt(np.mean(np.square(fold_persistence_errors))),
            )
        )
        fold_coefficients.append(np.asarray(model.named_steps["ridge"].coef_, dtype=float))
    folds = len(fold_coefficients)
    if not folds:
        return None, None
    environmental_rmse = float(np.sqrt(np.mean(np.square(model_errors))))
    seasonal_rmse = float(np.sqrt(np.mean(np.square(seasonal_errors))))
    persistence_rmse = float(np.sqrt(np.mean(np.square(persistence_errors))))
    skill_seasonal = float(1 - environmental_rmse / seasonal_rmse) if seasonal_rmse > 0 else 0.0
    skill_persistence = float(1 - environmental_rmse / persistence_rmse) if persistence_rmse > 0 else 0.0
    win_rate = float(np.mean(fold_wins))
    final_model = template.fit(frame[features], frame[target])
    top_effects, stability = _top_effects(final_model, features, fold_coefficients)
    active = (
        folds >= MINIMUM_ACTIVATION_FOLDS
        and skill_seasonal > MINIMUM_SKILL
        and skill_persistence > MINIMUM_SKILL
        and win_rate >= MINIMUM_FOLD_WIN_RATE
        and stability >= MINIMUM_EFFECT_STABILITY
    )
    validation = OutcomeValidation(
        species=str(identity.species), population=str(identity.population), season=str(identity.season),
        outcome=outcome, outcome_kind="continuous", unit=unit,
        years=int(frame.year.nunique()), folds=folds,
        seasonal_error=seasonal_rmse, persistence_error=persistence_rmse,
        environmental_error=environmental_rmse,
        skill_vs_seasonal=skill_seasonal, skill_vs_persistence=skill_persistence,
        fold_win_rate=win_rate, effect_stability=stability,
        environment_coverage=coverage,
        status="active historical association" if active else "not activated",
        feature_columns=tuple(features), top_effects=top_effects,
    )
    return final_model, validation


def _corridor_model(frame: pd.DataFrame, features: list[str], identity, coverage: float):
    if frame.corridor_choice.nunique() < 2:
        return None, None
    model_errors, seasonal_errors, persistence_errors = [], [], []
    valid_folds = 0
    fold_wins = []
    years = sorted(frame.year.unique())
    for test_year in years[MINIMUM_TRAIN_YEARS:]:
        train, test = frame[frame.year < test_year], frame[frame.year == test_year]
        if train.corridor_choice.nunique() < 2:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
            LogisticRegression(C=0.5, max_iter=1000, random_state=41),
        ).fit(train[features], train.corridor_choice)
        actual = test.corridor_choice.astype(str).to_numpy()
        fold_model_errors = (model.predict(test[features]).astype(str) != actual).astype(int)
        seasonal = str(train.corridor_choice.mode().iloc[0])
        previous_values = train[train.year.eq(train.year.max())].corridor_choice
        previous = str(previous_values.mode().iloc[0])
        fold_seasonal_errors = (np.repeat(seasonal, len(test)) != actual).astype(int)
        fold_persistence_errors = (np.repeat(previous, len(test)) != actual).astype(int)
        model_errors.extend(fold_model_errors)
        seasonal_errors.extend(fold_seasonal_errors)
        persistence_errors.extend(fold_persistence_errors)
        fold_wins.append(
            float(np.mean(fold_model_errors))
            <= min(float(np.mean(fold_seasonal_errors)), float(np.mean(fold_persistence_errors)))
        )
        valid_folds += 1
    if not valid_folds:
        return None, None
    environmental_error = float(np.mean(model_errors))
    seasonal_error = float(np.mean(seasonal_errors))
    persistence_error = float(np.mean(persistence_errors))
    skill_seasonal = seasonal_error - environmental_error
    skill_persistence = persistence_error - environmental_error
    win_rate = float(np.mean(fold_wins))
    final_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
        LogisticRegression(C=0.5, max_iter=1000, random_state=41),
    ).fit(frame[features], frame.corridor_choice)
    active = (
        valid_folds >= MINIMUM_ACTIVATION_FOLDS
        and skill_seasonal > MINIMUM_SKILL
        and skill_persistence > MINIMUM_SKILL
        and win_rate >= MINIMUM_FOLD_WIN_RATE
    )
    validation = OutcomeValidation(
        species=str(identity.species), population=str(identity.population), season=str(identity.season),
        outcome="corridor_choice", outcome_kind="classification", unit="error rate",
        years=int(frame.year.nunique()), folds=valid_folds,
        seasonal_error=seasonal_error, persistence_error=persistence_error,
        environmental_error=environmental_error,
        skill_vs_seasonal=skill_seasonal, skill_vs_persistence=skill_persistence,
        fold_win_rate=win_rate, effect_stability=1.0,
        environment_coverage=coverage,
        status="active historical association" if active else "not activated",
        feature_columns=tuple(features), top_effects=tuple(),
    )
    return final_model, validation


def fit_outcome_models(telemetry: pd.DataFrame, environment: pd.DataFrame):
    frame = outcome_training_frame(telemetry, environment)
    if frame.empty:
        raise ValueError("No population-year outcomes overlap environmental history.")
    if frame.year.nunique() < MINIMUM_TRAIN_YEARS + 1:
        raise ValueError("At least four matched years are required for a rolling evaluation.")
    identity = frame.iloc[0]
    all_features, all_coverage = _usable_features(frame)
    if not all_features:
        raise ValueError(
            "No environmental feature has at least 35% multi-year coverage and variation."
        )
    models, validations = {}, []
    source_trials: list[dict] = []
    retained_sources: dict[str, list[str]] = {}
    evidence_sources: dict[str, list[str]] = {}
    for outcome, (target, unit) in CONTINUOUS_OUTCOMES.items():
        features, coverage = _usable_features(frame, OUTCOME_WINDOWS[outcome])
        if not features:
            features, coverage = all_features, all_coverage
        accepted_sources = []
        for source, source_features in _source_feature_groups(features).items():
            _, trial = _continuous_model(
                frame.dropna(subset=[target]).reset_index(drop=True),
                source_features, target, identity, outcome, unit,
                float(frame[source_features].notna().mean().mean()),
            )
            if trial is None:
                continue
            source_trials.append(
                {
                    **asdict(trial),
                    "source_group": source,
                    "model_status": "included",
                    "evidence_status": trial.status,
                    "status": "included",
                }
            )
            if _accepted(trial):
                accepted_sources.append(source)
        retained = [
            feature for feature in features if _feature_source(feature) in accepted_sources
        ]
        # Every sufficiently covered source is active in the fitted classroom model.
        # Independent source trials remain visible as evidence diagnostics and do
        # not masquerade as proof that each source improves unseen-year skill alone.
        final_features = features
        model, validation = _continuous_model(
            frame.dropna(subset=[target]).reset_index(drop=True),
            final_features, target, identity, outcome, unit, coverage,
        )
        if model is not None:
            models[outcome] = model
            validations.append(validation)
            retained_sources[outcome] = sorted(_source_feature_groups(features))
            evidence_sources[outcome] = sorted(accepted_sources)
    corridor_features, corridor_coverage = _usable_features(frame, OUTCOME_WINDOWS["corridor_choice"])
    if not corridor_features:
        corridor_features, corridor_coverage = all_features, all_coverage
    accepted_corridor_sources = []
    for source, source_features in _source_feature_groups(corridor_features).items():
        _, trial = _corridor_model(
            frame, source_features, identity,
            float(frame[source_features].notna().mean().mean()),
        )
        if trial is None:
            continue
        source_trials.append(
            {
                **asdict(trial),
                "source_group": source,
                "model_status": "included",
                "evidence_status": trial.status,
                "status": "included",
            }
        )
        if _accepted(trial):
            accepted_corridor_sources.append(source)
    model, validation = _corridor_model(
        frame, corridor_features, identity, corridor_coverage
    )
    if model is not None:
        models["corridor_choice"] = model
        validations.append(validation)
        retained_sources["corridor_choice"] = sorted(
            _source_feature_groups(corridor_features)
        )
        evidence_sources["corridor_choice"] = sorted(accepted_corridor_sources)
    features = sorted(set(feature for model in models.values() for feature in model.feature_names_in_))
    reference = {column: float(pd.to_numeric(frame[column], errors="coerce").median()) for column in features}
    bearings = {}
    if "window_name" in environment and "route_bearing_degrees" in environment:
        bearings = environment.groupby("window_name").route_bearing_degrees.median().astype(float).to_dict()
    bundle = {
        "schema_version": 4,
        "species": str(identity.species),
        "population": str(identity.population),
        "season": str(identity.season),
        "features": features,
        "reference_environment": reference,
        "reference_bearings": bearings,
        "models": models,
        "validations": [asdict(validation) for validation in validations],
        "source_trials": source_trials,
        "retained_sources": retained_sources,
        "evidence_sources": evidence_sources,
    }
    return bundle, validations


def _adjust_vector(reference: dict, changed: dict, u: str, v: str, delta: float, direction):
    if u not in reference or v not in reference:
        return
    if float(delta) == 0.0 and direction is None:
        return
    observed_u, observed_v = reference[u], reference[v]
    speed = float(np.hypot(observed_u, observed_v))
    angle = np.arctan2(observed_u, observed_v) if direction is None else np.radians(float(direction))
    target = max(0.0, speed + float(delta))
    changed[u], changed[v] = target * np.sin(angle), target * np.cos(angle)


def predict_activated_effects(bundle: dict, scenario: WeatherScenario) -> dict[str, dict]:
    reference = dict(bundle.get("reference_environment", {}))
    changed = dict(reference)
    for column in changed:
        if column == "temperature_2m" or "__temperature_2m_" in column:
            changed[column] += scenario.temperature_change_c
        elif column == "surface_pressure" or column.endswith("__surface_pressure_mean"):
            changed[column] += scenario.pressure_trend_hpa_per_day * 3
        elif "__wind_speed_10m_" in column or column.endswith("__wind_gusts_10m_max"):
            changed[column] = max(0.0, changed[column] + scenario.wind_speed_change_kmh)
    prefixes = {column.rsplit("__", 1)[0] for column in changed if column.endswith("__wind_u_component_10m")}
    for prefix in prefixes:
        if f"{prefix}__wind_v_component_10m" not in changed:
            continue
        if scenario.wind_speed_change_kmh == 0 and scenario.wind_direction_deg is None:
            continue
        _adjust_vector(
            reference, changed, f"{prefix}__wind_u_component_10m", f"{prefix}__wind_v_component_10m",
            scenario.wind_speed_change_kmh, scenario.wind_direction_deg,
        )
        u, v = changed[f"{prefix}__wind_u_component_10m"], changed[f"{prefix}__wind_v_component_10m"]
        bearing = np.radians(float(bundle.get("reference_bearings", {}).get(prefix, 0.0)))
        if f"{prefix}__tailwind_10m_mean" in changed:
            changed[f"{prefix}__tailwind_10m_mean"] = u * np.sin(bearing) + v * np.cos(bearing)
        if f"{prefix}__crosswind_10m_mean" in changed:
            changed[f"{prefix}__crosswind_10m_mean"] = u * np.cos(bearing) - v * np.sin(bearing)
    _adjust_vector(reference, changed, "wind_u_component_10m", "wind_v_component_10m", scenario.wind_speed_change_kmh, scenario.wind_direction_deg)
    _adjust_vector(reference, changed, "marine_current_u", "marine_current_v", scenario.current_speed_change_kmh, scenario.current_direction_deg)
    for column in changed:
        if column.endswith("__tailwind_10m_mean") or column.endswith(
            "__flight_hour_tailwind_mean"
        ):
            changed[column] += scenario.tailwind_change_kmh
        elif column.endswith("__flight_hour_wind_speed_mean"):
            changed[column] = max(
                0.0, changed[column] + abs(scenario.tailwind_change_kmh)
            )
        elif column.endswith("__thermal_uplift_proxy"):
            changed[column] = max(
                0.0, changed[column] * (1.0 + scenario.uplift_change_fraction)
            )
        elif column.endswith("__sea_ice_concentration"):
            changed[column] = float(np.clip(
                changed[column] + scenario.sea_ice_concentration_change, 0.0, 1.0
            ))
        elif column.endswith("__distance_to_sea_ice_km"):
            changed[column] = max(
                0.0, changed[column] + scenario.sea_ice_distance_change_km
            )
    validation_by_outcome = {row["outcome"]: row for row in bundle.get("validations", [])}
    effects = {}
    for outcome, model in bundle.get("models", {}).items():
        validation = validation_by_outcome.get(outcome, {})
        if validation.get("status") != "active historical association":
            continue
        features = list(model.feature_names_in_)
        x0 = pd.DataFrame([[reference[column] for column in features]], columns=features)
        x1 = pd.DataFrame([[changed[column] for column in features]], columns=features)
        if validation.get("outcome_kind") == "classification":
            baseline = dict(zip(model.classes_, model.predict_proba(x0)[0].astype(float)))
            simulated = dict(zip(model.classes_, model.predict_proba(x1)[0].astype(float)))
            if max(
                abs(float(simulated[label]) - float(baseline[label]))
                for label in baseline
            ) < MINIMUM_SCENARIO_PROBABILITY_CHANGE:
                continue
            effects[outcome] = {
                "kind": "classification",
                "baseline": baseline,
                "scenario": simulated,
            }
        else:
            baseline_prediction = float(model.predict(x0)[0])
            scenario_prediction = float(model.predict(x1)[0])
            if outcome in {"migration_duration", "migration_pace", "stopovers"}:
                baseline_prediction = max(0.0, baseline_prediction)
                scenario_prediction = max(0.0, scenario_prediction)
            delta = scenario_prediction - baseline_prediction
            if abs(delta) < 1e-9:
                continue
            effects[outcome] = {
                "kind": "continuous",
                "baseline": baseline_prediction,
                "scenario": scenario_prediction,
                "delta": delta,
                "unit": validation.get("unit", ""),
            }
    return effects


def outcome_model_key(species: str, population: str, season: str) -> str:
    return "__".join([slug(species), slug(population), slug(season)])


def save_outcome_models(
    bundle: dict,
    validations: list[OutcomeValidation],
    catalog_root: Path | str = DEFAULT_CATALOG_ROOT,
) -> tuple[Path, Path]:
    root = Path(catalog_root) / "outcome-models"
    root.mkdir(parents=True, exist_ok=True)
    key = outcome_model_key(bundle["species"], bundle["population"], bundle["season"])
    model_path, metrics_path = root / f"{key}.joblib", root / f"{key}.json"
    temporary_model, temporary_metrics = root / f".{key}.joblib.tmp", root / f".{key}.json.tmp"
    joblib.dump(bundle, temporary_model)
    temporary_metrics.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "species": bundle["species"], "population": bundle["population"],
                "season": bundle["season"],
                "validations": [asdict(validation) for validation in validations],
                "source_trials": bundle.get("source_trials", []),
                "retained_sources": bundle.get("retained_sources", {}),
                "evidence_sources": bundle.get("evidence_sources", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_model.replace(model_path)
    temporary_metrics.replace(metrics_path)
    return model_path, metrics_path
