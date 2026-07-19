from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from .catalog import DEFAULT_CATALOG_ROOT, slug
from .population import build_population_corridor


HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
ATMOSPHERE_FIELDS = (
    "temperature_2m,surface_pressure,precipitation,wind_speed_10m,wind_direction_10m"
)
DAILY_ATMOSPHERE_FIELDS = (
    "temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
    "relative_humidity_2m_mean,surface_pressure_mean,precipitation_sum,snowfall_sum,"
    "daylight_duration,sunshine_duration,wind_speed_10m_mean,wind_speed_10m_max,"
    "wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,"
    "et0_fao_evapotranspiration"
)
JOURNEY_ENVIRONMENT_COLUMNS = [
    "species", "population", "movement_type", "season", "year", "animal_id",
    "journey_id", "window_name", "anchor_date", "start_date", "end_date",
    "latitude", "longitude", "route_bearing_degrees", "days_requested", "days_observed",
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "relative_humidity_2m_mean", "surface_pressure_mean", "precipitation_sum",
    "precipitation_days", "snowfall_sum", "daylight_duration_hours",
    "sunshine_duration_hours", "wind_speed_10m_mean", "wind_speed_10m_max",
    "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "wind_u_component_10m", "wind_v_component_10m", "tailwind_10m_mean",
    "crosswind_10m_mean", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "sea_surface_temperature", "marine_current_u", "marine_current_v",
    "marine_current_tailwind", "mixed_layer_depth", "sea_surface_height",
    "sea_surface_salinity", "chlorophyll_a", "primary_productivity",
    "bathymetry_m", "sst_front_gradient", "lemming_density_index",
    "sea_ice_concentration", "sea_ice_concentration_stdev", "distance_to_sea_ice_km", "snow_depth_mean",
    "snow_depth_max", "soil_temperature_mean", "boundary_layer_height_mean",
    "boundary_layer_height_max", "cape_mean", "cloud_cover_mean",
    "direct_radiation_mean", "diffuse_radiation_mean",
    "flight_hour_wind_speed_mean", "flight_hour_tailwind_mean",
    "flight_hour_crosswind_mean", "thermal_uplift_proxy",
    "hourly_source", "lemming_source", "sea_ice_source", "glorys_source",
    "marine_biology_source",
    "environment_source",
]
ENVIRONMENT_COLUMNS = [
    "species",
    "population",
    "movement_type",
    "season",
    "year",
    "animal_id",
    "progress_bin",
    "date",
    "latitude",
    "longitude",
    "temperature_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_u_component_10m",
    "wind_v_component_10m",
    "sea_surface_temperature",
    "marine_current_u",
    "marine_current_v",
    "environment_source",
]


class EnvironmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnvironmentRefreshResult:
    rows: int
    atmosphere_rows: int
    marine_rows: int
    years: int
    source: str


def build_sampling_points(telemetry: pd.DataFrame, bins: int = 12) -> pd.DataFrame:
    """Represent each population-season-year corridor with sparse route locations."""
    required = {
        "species",
        "population",
        "movement_type",
        "season",
        "year",
        "animal_id",
        "timestamp_utc",
        "latitude",
        "longitude",
    }
    if not required.issubset(telemetry):
        raise EnvironmentError(f"Missing telemetry columns: {sorted(required-set(telemetry))}")
    rows = []
    group_columns = [
        "species",
        "population",
        "movement_type",
        "season",
        "year",
    ]
    for keys, group in telemetry.groupby(group_columns, sort=True):
        try:
            population = build_population_corridor(group, bins=bins, minimum_points=2)
        except ValueError:
            continue
        sampled = population.corridor.copy()
        median_times = (
            population.paths.assign(timestamp_ns=lambda value: value.timestamp_utc.astype("int64"))
            .groupby("progress", sort=True)
            .timestamp_ns.median()
        )
        sampled["timestamp_utc"] = pd.to_datetime(
            sampled.progress.map(median_times).astype("int64"), utc=True
        )
        metadata = dict(zip(group_columns, keys))
        for progress_bin, point in sampled.reset_index(drop=True).iterrows():
            rows.append(
                {
                    **metadata,
                    "animal_id": "population:" + slug(metadata["population"]),
                    "progress_bin": int(progress_bin),
                    "date": pd.Timestamp(point.timestamp_utc).date(),
                    "latitude": round(float(point.latitude), 2),
                    "longitude": round(float(point.longitude), 2),
                }
            )
    return pd.DataFrame(rows)


def _bearing_degrees(start_latitude, start_longitude, end_latitude, end_longitude) -> float:
    lat1, lat2 = np.radians([float(start_latitude), float(end_latitude)])
    delta = np.radians(float(end_longitude) - float(start_longitude))
    east = np.sin(delta) * np.cos(lat2)
    north = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta)
    return float((np.degrees(np.arctan2(east, north)) + 360) % 360)


def _interpolated_location(ordered: pd.DataFrame, progress: float) -> tuple[pd.Timestamp, float, float]:
    # Pandas/Arrow may preserve telemetry at microsecond precision; Timestamp.value
    # normalizes every point to nanoseconds before interpolation.
    times = np.asarray([pd.Timestamp(value).value for value in ordered.timestamp_utc], dtype=float)
    target = times[0] + (times[-1] - times[0]) * progress
    longitude = np.degrees(np.unwrap(np.radians(ordered.longitude.to_numpy(float))))
    timestamp = pd.to_datetime(int(target), utc=True)
    latitude = float(np.interp(target, times, ordered.latitude.to_numpy(float)))
    longitude_value = float(((np.interp(target, times, longitude) + 180) % 360) - 180)
    return timestamp, latitude, longitude_value


def build_journey_weather_points(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Create outcome-specific historical weather windows for every observed journey."""
    required = {
        "species", "population", "movement_type", "season", "year", "animal_id",
        "timestamp_utc", "latitude", "longitude",
    }
    if not required.issubset(telemetry):
        raise EnvironmentError(f"Missing telemetry columns: {sorted(required-set(telemetry))}")
    frame = telemetry.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    frame["journey_id"] = (
        frame.animal_id.astype(str) + "|" + frame.year.astype(str) + "|" + frame.season.astype(str)
    )
    rows = []
    for journey_id, group in frame.groupby("journey_id", sort=True):
        ordered = group.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
        if len(ordered) < 4 or ordered.timestamp_utc.iloc[-1] <= ordered.timestamp_utc.iloc[0]:
            continue
        first, last = ordered.iloc[0], ordered.iloc[-1]
        bearing = _bearing_degrees(first.latitude, first.longitude, last.latitude, last.longitude)
        windows = [
            ("departure_30d", first.timestamp_utc, first.latitude, first.longitude, -30, -1),
            ("departure_7d", first.timestamp_utc, first.latitude, first.longitude, -7, -1),
        ]
        for name, progress in (("route_early_7d", 0.25), ("route_middle_7d", 0.50), ("route_late_7d", 0.75)):
            timestamp, latitude, longitude = _interpolated_location(ordered, progress)
            windows.append((name, timestamp, latitude, longitude, -3, 3))
        windows.append(("arrival_14d", last.timestamp_utc, last.latitude, last.longitude, -13, 0))
        identity = first
        for name, anchor, latitude, longitude, before, after in windows:
            anchor = pd.Timestamp(anchor)
            start_date = (anchor + pd.Timedelta(days=before)).date()
            end_date = (anchor + pd.Timedelta(days=after)).date()
            rows.append(
                {
                    "species": str(identity.species),
                    "population": str(identity.population),
                    "movement_type": str(identity.movement_type),
                    "season": str(identity.season),
                    "year": int(identity.year),
                    "animal_id": str(identity.animal_id),
                    "journey_id": str(journey_id),
                    "window_name": name,
                    "anchor_date": anchor.date(),
                    "start_date": start_date,
                    "end_date": end_date,
                    "latitude": round(float(latitude), 3),
                    "longitude": round(float(longitude), 3),
                    "route_bearing_degrees": bearing,
                    "days_requested": int((end_date - start_date).days + 1),
                    "request_month": str(start_date)[:7],
                }
            )
    return pd.DataFrame(rows)


def _request_json(url: str, params: dict, timeout: int = 60):
    last = None
    for attempt in range(6):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt < 5:
                retry_after = None
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    retry_after = exc.response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else min(5.0 * (2**attempt), 60.0)
                except ValueError:
                    delay = min(5.0 * (2**attempt), 60.0)
                time.sleep(max(1.0, delay))
    raise EnvironmentError(f"Environmental history request failed: {last}")


def _mean_direction(values) -> float:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return float("nan")
    angles = np.radians(values)
    return float((np.degrees(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())) + 360) % 360)


def _atmosphere_payload(keys: pd.DataFrame) -> list[dict]:
    start_date = str(pd.to_datetime(keys.date).min().date())
    end_date = str(pd.to_datetime(keys.date).max().date())
    payload = _request_json(
        HISTORICAL_WEATHER_URL,
        {
            "latitude": ",".join(keys.latitude.astype(str)),
            "longitude": ",".join(keys.longitude.astype(str)),
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ATMOSPHERE_FIELDS,
            "models": "era5",
            "timezone": "UTC",
        },
    )
    return payload if isinstance(payload, list) else [payload]


def fetch_atmosphere_history(points: pd.DataFrame, batch_size: int = 25) -> pd.DataFrame:
    """Fetch matched ERA5 history at sparse route nodes across all observed years."""
    if points.empty:
        return pd.DataFrame(columns=ENVIRONMENT_COLUMNS)
    rows = []
    population_year = ["species", "population", "season", "year"]
    for _, same_year in points.groupby(population_year, sort=True):
        same_year = same_year.reset_index(drop=True)
        for start in range(0, len(same_year), batch_size):
            batch = same_year.iloc[start : start + batch_size]
            responses = _atmosphere_payload(batch)
            for (_, point), response in zip(batch.iterrows(), responses):
                hourly = response.get("hourly", {})
                times = pd.to_datetime(hourly.get("time", []), utc=True, errors="coerce")
                wanted_date = pd.Timestamp(point.date).date()
                positions = np.array([value.date() == wanted_date for value in times])
                temperature = pd.to_numeric(
                    pd.Series(hourly.get("temperature_2m", [])), errors="coerce"
                )[positions]
                pressure = pd.to_numeric(
                    pd.Series(hourly.get("surface_pressure", [])), errors="coerce"
                )[positions]
                precipitation = pd.to_numeric(
                    pd.Series(hourly.get("precipitation", [])), errors="coerce"
                )[positions]
                wind_speed = pd.to_numeric(
                    pd.Series(hourly.get("wind_speed_10m", [])), errors="coerce"
                )[positions]
                directions = pd.Series(hourly.get("wind_direction_10m", []))[positions]
                direction = _mean_direction(directions)
                speed = float(wind_speed.mean()) if wind_speed.notna().any() else float("nan")
                east = -speed * np.sin(np.radians(direction))
                north = -speed * np.cos(np.radians(direction))
                rows.append(
                    {
                        **point.to_dict(),
                        "temperature_2m": float(temperature.mean()),
                        "surface_pressure": float(pressure.mean()),
                        "precipitation": float(precipitation.sum()),
                        "wind_speed_10m": speed,
                        "wind_direction_10m": direction,
                        "wind_u_component_10m": float(east),
                        "wind_v_component_10m": float(north),
                        "sea_surface_temperature": np.nan,
                        "marine_current_u": np.nan,
                        "marine_current_v": np.nan,
                        "environment_source": "Open-Meteo ERA5 reanalysis",
                    }
                )
    return pd.DataFrame(rows).reindex(columns=ENVIRONMENT_COLUMNS)


def _daily_atmosphere_payload(keys: pd.DataFrame) -> list[dict]:
    api_key = os.getenv("OPEN_METEO_API_KEY", "").strip()
    url = (
        "https://customer-archive-api.open-meteo.com/v1/archive"
        if api_key else HISTORICAL_WEATHER_URL
    )
    parameters = {
        "latitude": ",".join(keys.latitude.astype(str)),
        "longitude": ",".join(keys.longitude.astype(str)),
        "start_date": str(pd.to_datetime(keys.start_date).min().date()),
        "end_date": str(pd.to_datetime(keys.end_date).max().date()),
        "daily": DAILY_ATMOSPHERE_FIELDS,
        "models": "era5",
        "timezone": "UTC",
        "cell_selection": (
            "sea" if keys.movement_type.eq("marine").all()
            else "land" if keys.movement_type.eq("terrestrial").all()
            else "nearest"
        ),
    }
    if api_key:
        parameters["apikey"] = api_key
    payload = _request_json(
        url,
        parameters,
        timeout=90,
    )
    return payload if isinstance(payload, list) else [payload]


def _numeric_daily(daily: dict, name: str, positions: np.ndarray) -> pd.Series:
    values = pd.to_numeric(pd.Series(daily.get(name, [])), errors="coerce")
    if len(values) != len(positions):
        return pd.Series(dtype=float)
    return values[positions]


def _aggregate_journey_weather(point: pd.Series, response: dict) -> dict:
    daily = response.get("daily", {})
    dates = pd.to_datetime(daily.get("time", []), errors="coerce")
    start, end = pd.Timestamp(point.start_date), pd.Timestamp(point.end_date)
    positions = np.asarray((dates >= start) & (dates <= end))
    speed = _numeric_daily(daily, "wind_speed_10m_mean", positions)
    directions = _numeric_daily(daily, "wind_direction_10m_dominant", positions)
    valid = speed.notna() & directions.notna()
    if valid.any():
        radians = np.radians(directions[valid].to_numpy(float))
        # Meteorological direction is where wind comes from; convert to motion toward east/north.
        east_values = -speed[valid].to_numpy(float) * np.sin(radians)
        north_values = -speed[valid].to_numpy(float) * np.cos(radians)
        east, north = float(np.mean(east_values)), float(np.mean(north_values))
        direction = float((np.degrees(np.arctan2(-east, -north)) + 360) % 360)
    else:
        east = north = direction = float("nan")
    bearing = np.radians(float(point.route_bearing_degrees))
    tailwind = east * np.sin(bearing) + north * np.cos(bearing)
    crosswind = east * np.cos(bearing) - north * np.sin(bearing)

    def mean(name):
        values = _numeric_daily(daily, name, positions)
        return float(values.mean()) if values.notna().any() else float("nan")

    def maximum(name):
        values = _numeric_daily(daily, name, positions)
        return float(values.max()) if values.notna().any() else float("nan")

    def total(name):
        values = _numeric_daily(daily, name, positions)
        return float(values.sum(min_count=1)) if values.notna().any() else float("nan")

    precipitation = _numeric_daily(daily, "precipitation_sum", positions)
    return {
        **{column: point[column] for column in (
            "species", "population", "movement_type", "season", "year", "animal_id",
            "journey_id", "window_name", "anchor_date", "start_date", "end_date", "latitude",
            "longitude", "route_bearing_degrees", "days_requested",
        )},
        "days_observed": int(positions.sum()),
        "temperature_2m_mean": mean("temperature_2m_mean"),
        "temperature_2m_max": maximum("temperature_2m_max"),
        "temperature_2m_min": float(_numeric_daily(daily, "temperature_2m_min", positions).min()),
        "relative_humidity_2m_mean": mean("relative_humidity_2m_mean"),
        "surface_pressure_mean": mean("surface_pressure_mean"),
        "precipitation_sum": total("precipitation_sum"),
        "precipitation_days": int((precipitation >= 1.0).sum()),
        "snowfall_sum": total("snowfall_sum"),
        "daylight_duration_hours": mean("daylight_duration") / 3600,
        "sunshine_duration_hours": mean("sunshine_duration") / 3600,
        "wind_speed_10m_mean": mean("wind_speed_10m_mean"),
        "wind_speed_10m_max": maximum("wind_speed_10m_max"),
        "wind_gusts_10m_max": maximum("wind_gusts_10m_max"),
        "wind_direction_10m_dominant": direction,
        "wind_u_component_10m": east,
        "wind_v_component_10m": north,
        "tailwind_10m_mean": float(tailwind),
        "crosswind_10m_mean": float(crosswind),
        "shortwave_radiation_sum": total("shortwave_radiation_sum"),
        "et0_fao_evapotranspiration": total("et0_fao_evapotranspiration"),
        "sea_surface_temperature": np.nan,
        "marine_current_u": np.nan,
        "marine_current_v": np.nan,
        "environment_source": "Open-Meteo ERA5 daily reanalysis",
    }


def fetch_journey_atmosphere_history(points: pd.DataFrame, batch_size: int = 25) -> pd.DataFrame:
    """Fetch daily ERA5 summaries for individual migration outcome windows."""
    if points.empty:
        return pd.DataFrame(columns=JOURNEY_ENVIRONMENT_COLUMNS)
    rows = []
    # Month-bounded batches keep payloads compact while allowing different biological
    # windows to share one multi-location API request.
    groups = ["species", "population", "movement_type", "season", "year", "request_month"]
    for _, group in points.groupby(groups, sort=True):
        group = group.reset_index(drop=True)
        for start in range(0, len(group), batch_size):
            batch = group.iloc[start : start + batch_size]
            responses = _daily_atmosphere_payload(batch)
            if len(responses) != len(batch):
                raise EnvironmentError(
                    f"Weather service returned {len(responses)} locations for a {len(batch)}-location request."
                )
            rows.extend(
                _aggregate_journey_weather(point, response)
                for (_, point), response in zip(batch.iterrows(), responses)
            )
            # Multi-coordinate locations count individually toward the public API's
            # rate budget. A steady pace is much more reliable than burst-and-retry.
            time.sleep(max(1.0, len(batch) / 8.0))
    return pd.DataFrame(rows).reindex(columns=JOURNEY_ENVIRONMENT_COLUMNS)


def fetch_copernicus_marine_history(
    points: pd.DataFrame,
    dataset_id: str = "cmems_mod_glo_phy_my_0.083deg_P1D-m",
) -> pd.DataFrame:
    """Sample daily GLORYS reanalysis. Requires the optional copernicusmarine package/login."""
    marine_points = points[points.movement_type.eq("marine")].copy()
    if marine_points.empty:
        return pd.DataFrame(columns=ENVIRONMENT_COLUMNS)
    try:
        import copernicusmarine
    except ImportError as exc:
        raise EnvironmentError(
            "Install requirements-marine.txt and run `copernicusmarine login` before marine refresh."
        ) from exc
    minimum_date = pd.to_datetime(marine_points.date).min()
    maximum_date = pd.to_datetime(marine_points.date).max() + pd.Timedelta(days=1)
    try:
        dataset = copernicusmarine.open_dataset(
            dataset_id=dataset_id,
            variables=["uo", "vo", "thetao"],
            minimum_longitude=float(marine_points.longitude.min()) - 0.2,
            maximum_longitude=float(marine_points.longitude.max()) + 0.2,
            minimum_latitude=float(marine_points.latitude.min()) - 0.2,
            maximum_latitude=float(marine_points.latitude.max()) + 0.2,
            start_datetime=minimum_date.isoformat(),
            end_datetime=maximum_date.isoformat(),
        )
    except Exception as exc:
        raise EnvironmentError(f"Copernicus Marine dataset could not be opened: {exc}") from exc
    if "depth" in dataset.coords:
        dataset = dataset.sel(depth=0, method="nearest")
    rows = []
    for _, point in marine_points.iterrows():
        try:
            selected = dataset.sel(
                time=pd.Timestamp(point.date),
                latitude=float(point.latitude),
                longitude=float(point.longitude),
                method="nearest",
            )
            rows.append(
                {
                    **point.to_dict(),
                    "temperature_2m": np.nan,
                    "surface_pressure": np.nan,
                    "precipitation": np.nan,
                    "wind_speed_10m": np.nan,
                    "wind_direction_10m": np.nan,
                    "wind_u_component_10m": np.nan,
                    "wind_v_component_10m": np.nan,
                    "sea_surface_temperature": float(selected.thetao.values),
                    "marine_current_u": float(selected.uo.values) * 3.6,
                    "marine_current_v": float(selected.vo.values) * 3.6,
                    "environment_source": "Copernicus Marine GLORYS reanalysis",
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(rows).reindex(columns=ENVIRONMENT_COLUMNS)


def merge_environment(atmosphere: pd.DataFrame, marine: pd.DataFrame) -> pd.DataFrame:
    if atmosphere.empty:
        return marine.copy()
    if marine.empty:
        return atmosphere.copy()
    keys = [
        "species",
        "population",
        "movement_type",
        "season",
        "year",
        "animal_id",
        "progress_bin",
        "date",
        "latitude",
        "longitude",
    ]
    combined = atmosphere.merge(
        marine[
            keys
            + ["sea_surface_temperature", "marine_current_u", "marine_current_v"]
        ],
        on=keys,
        how="left",
        suffixes=("", "_marine"),
    )
    for column in ["sea_surface_temperature", "marine_current_u", "marine_current_v"]:
        combined[column] = combined[f"{column}_marine"].combine_first(combined[column])
        combined.drop(columns=f"{column}_marine", inplace=True)
    combined.loc[combined.marine_current_u.notna(), "environment_source"] = (
        "Open-Meteo ERA5 + Copernicus Marine GLORYS reanalysis"
    )
    return combined.reindex(columns=ENVIRONMENT_COLUMNS)


def write_environment(
    frame: pd.DataFrame,
    catalog_root: Path | str = DEFAULT_CATALOG_ROOT,
    merge_existing: bool = True,
) -> Path:
    root = Path(catalog_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "environment.parquet"
    temporary = root / "environment.tmp.parquet"
    if merge_existing and path.exists() and not frame.empty:
        existing = pd.read_parquet(path)
        replacement_keys = ["species", "population", "season", "year"]
        replacements = frame[replacement_keys].drop_duplicates()
        retained = existing.merge(
            replacements.assign(_replace=True), on=replacement_keys, how="left"
        )
        retained = retained[retained._replace.isna()].drop(columns="_replace")
        frame = pd.concat([retained, frame], ignore_index=True)
        frame = frame.drop_duplicates(
            ["species", "population", "season", "year", "progress_bin"], keep="last"
        )
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)
    return path


def write_journey_environment(
    frame: pd.DataFrame,
    catalog_root: Path | str = DEFAULT_CATALOG_ROOT,
    merge_existing: bool = True,
) -> Path:
    """Atomically store journey windows, replacing only successfully refreshed journeys."""
    root = Path(catalog_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "environment-journeys.parquet"
    temporary = root / "environment-journeys.tmp.parquet"
    if merge_existing and path.exists() and not frame.empty:
        existing = pd.read_parquet(path)
        replacements = frame[["journey_id"]].drop_duplicates().assign(_replace=True)
        retained = existing.merge(replacements, on="journey_id", how="left")
        retained = retained[retained._replace.isna()].drop(columns="_replace")
        frame = pd.concat([retained, frame], ignore_index=True)
    frame = sanitize_journey_environment(frame)
    frame = frame.drop_duplicates(["species", "population", "season", "journey_id", "window_name"], keep="last")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)
    return path


def sanitize_journey_environment(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply physical range checks before predictors become model candidates."""
    output = frame.copy()
    ranges = {
        # ERA5-Land can expose glacier or ice-sheet thickness as apparent snow depth.
        "snow_depth_mean": (0.0, 5.0),
        "snow_depth_max": (0.0, 5.0),
        "sea_ice_concentration": (0.0, 1.0),
        "sea_ice_concentration_stdev": (0.0, 1.0),
        "distance_to_sea_ice_km": (0.0, 500.0),
        "chlorophyll_a": (0.0, None),
        "primary_productivity": (0.0, None),
        "boundary_layer_height_mean": (0.0, 10_000.0),
        "boundary_layer_height_max": (0.0, 10_000.0),
        "cloud_cover_mean": (0.0, 100.0),
    }
    for column, (minimum, maximum) in ranges.items():
        if column not in output:
            continue
        values = pd.to_numeric(output[column], errors="coerce")
        valid = values.ge(minimum)
        if maximum is not None:
            valid &= values.le(maximum)
        output[column] = values.where(valid)
    return output
