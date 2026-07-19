from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO, StringIO
from pathlib import Path
import math
import os
import time
from urllib.parse import quote
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

from .catalog import DEFAULT_CATALOG_ROOT
from .environmental_history import EnvironmentError, HISTORICAL_WEATHER_URL


JOIN_KEYS = ["journey_id", "window_name"]
BYLOT_ZENODO_URL = (
    "https://zenodo.org/api/records/16794619/files/"
    "BYLOT_species_abundance_dataset.zip/content"
)
BYLOT_SOURCE_DOI = "10.5281/zenodo.16794619"
ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"

HOURLY_FIELDS = (
    "temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,"
    "shortwave_radiation,direct_radiation,diffuse_radiation,wind_speed_10m,"
    "wind_direction_10m,wind_gusts_10m,boundary_layer_height,cape,"
    "snow_depth,snowfall,soil_temperature_0_to_7cm"
)

PREDICTOR_FEATURE_GROUPS = {
    "bylot_lemmings": {"lemming_density_index"},
    "nsidc_sea_ice": {
        "sea_ice_concentration", "sea_ice_concentration_stdev", "distance_to_sea_ice_km"
    },
    "era5_land_snow": {"snow_depth_mean", "snow_depth_max", "soil_temperature_mean"},
    "glorys": {
        "sea_surface_temperature",
        "marine_current_u",
        "marine_current_v",
        "marine_current_tailwind",
        "mixed_layer_depth",
        "sea_surface_height",
        "sea_surface_salinity",
    },
    "ocean_chlorophyll": {"chlorophyll_a"},
    "ocean_primary_productivity": {"primary_productivity"},
    "bathymetry": {"bathymetry_m"},
    "sst_fronts": {"sst_front_gradient"},
    "flight_conditions": {
        "boundary_layer_height_mean",
        "boundary_layer_height_max",
        "cape_mean",
        "cloud_cover_mean",
        "direct_radiation_mean",
        "diffuse_radiation_mean",
        "flight_hour_wind_speed_mean",
        "flight_hour_tailwind_mean",
        "flight_hour_crosswind_mean",
        "thermal_uplift_proxy",
    },
}

ENRICHMENT_COLUMNS = sorted(set().union(*PREDICTOR_FEATURE_GROUPS.values()))


@dataclass(frozen=True)
class SourceResult:
    source: str
    status: str
    rows: int
    message: str = ""


def _get(url: str, *, params: dict | None = None, timeout: int = 90) -> requests.Response:
    last: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "WildVector/1.0 classroom migration research"},
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            # A missing grid cell/date will not become available on retry. Only
            # throttling and server/network failures should consume the retry budget.
            if status is not None and 400 <= status < 500 and status != 429:
                break
            if attempt < 4:
                time.sleep(min(2**attempt, 20))
    raise EnvironmentError(f"Source request failed for {url}: {last}")


def _mean_direction(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(numeric) == 0:
        return float("nan")
    radians = np.radians(numeric)
    return float(
        (np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) + 360)
        % 360
    )


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else float("nan")


def _safe_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.max()) if numeric.notna().any() else float("nan")


def _hourly_payload(batch: pd.DataFrame) -> list[dict]:
    model = "era5_land" if batch.species.eq("Vulpes lagopus").all() else "era5"
    response = _get(
        HISTORICAL_WEATHER_URL,
        params={
            "latitude": ",".join(batch.latitude.astype(str)),
            "longitude": ",".join(batch.longitude.astype(str)),
            "start_date": str(pd.to_datetime(batch.start_date).min().date()),
            "end_date": str(pd.to_datetime(batch.end_date).max().date()),
            "hourly": HOURLY_FIELDS,
            "models": model,
            "timezone": "UTC",
            "cell_selection": "land",
        },
    ).json()
    return response if isinstance(response, list) else [response]


def _hourly_window_features(point: pd.Series, payload: dict) -> dict:
    hourly = payload.get("hourly", {})
    timestamps = pd.to_datetime(hourly.get("time", []), utc=True, errors="coerce")
    start = pd.Timestamp(point.start_date, tz="UTC")
    end = pd.Timestamp(point.end_date, tz="UTC") + pd.Timedelta(days=1)
    mask = np.asarray((timestamps >= start) & (timestamps < end))

    def series(name: str) -> pd.Series:
        values = pd.to_numeric(pd.Series(hourly.get(name, [])), errors="coerce")
        if len(values) != len(mask):
            return pd.Series(dtype=float)
        return values[mask].reset_index(drop=True)

    shortwave = series("shortwave_radiation")
    daylight = shortwave.fillna(0).gt(20)
    wind_speed = series("wind_speed_10m")
    wind_direction = series("wind_direction_10m")
    valid_wind = daylight & wind_speed.notna() & wind_direction.notna()
    if valid_wind.any():
        radians = np.radians(wind_direction[valid_wind].to_numpy(float))
        east = -wind_speed[valid_wind].to_numpy(float) * np.sin(radians)
        north = -wind_speed[valid_wind].to_numpy(float) * np.cos(radians)
        bearing = np.radians(float(point.route_bearing_degrees))
        tailwind = east * np.sin(bearing) + north * np.cos(bearing)
        crosswind = east * np.cos(bearing) - north * np.sin(bearing)
    else:
        tailwind = crosswind = np.asarray([], dtype=float)

    boundary = series("boundary_layer_height")
    cloud = series("cloud_cover")
    # This is deliberately named a proxy: it represents solar forcing multiplied by
    # boundary-layer development and cloud attenuation, not measured vertical velocity.
    common = shortwave.notna() & boundary.notna() & cloud.notna()
    uplift = (
        shortwave[common].clip(lower=0).to_numpy(float)
        / 1000.0
        * np.sqrt(boundary[common].clip(lower=0).to_numpy(float) / 1000.0)
        * (1 - cloud[common].clip(0, 100).to_numpy(float) / 100.0)
    )
    snow_depth = series("snow_depth")
    return {
        **{key: point[key] for key in JOIN_KEYS},
        "snow_depth_mean": _safe_mean(snow_depth),
        "snow_depth_max": _safe_max(snow_depth),
        "soil_temperature_mean": _safe_mean(series("soil_temperature_0_to_7cm")),
        "boundary_layer_height_mean": _safe_mean(boundary[daylight]),
        "boundary_layer_height_max": _safe_max(boundary[daylight]),
        "cape_mean": _safe_mean(series("cape")[daylight]),
        "cloud_cover_mean": _safe_mean(cloud[daylight]),
        "direct_radiation_mean": _safe_mean(series("direct_radiation")[daylight]),
        "diffuse_radiation_mean": _safe_mean(series("diffuse_radiation")[daylight]),
        "flight_hour_wind_speed_mean": _safe_mean(wind_speed[valid_wind]),
        "flight_hour_tailwind_mean": float(np.mean(tailwind)) if len(tailwind) else float("nan"),
        "flight_hour_crosswind_mean": float(np.mean(crosswind)) if len(crosswind) else float("nan"),
        "thermal_uplift_proxy": float(np.mean(uplift)) if len(uplift) else float("nan"),
        "hourly_source": "Open-Meteo ERA5 hourly reanalysis",
    }


def fetch_hourly_land_features(points: pd.DataFrame, batch_size: int = 10) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(columns=JOIN_KEYS + ENRICHMENT_COLUMNS + ["hourly_source"])
    rows: list[dict] = []
    work = points.copy()
    work["request_month"] = work.start_date.astype(str).str[:7]
    for _, group in work.groupby(["species", "season", "year", "request_month"], sort=True):
        group = group.reset_index(drop=True)
        for start in range(0, len(group), batch_size):
            batch = group.iloc[start : start + batch_size]
            payloads = _hourly_payload(batch)
            if len(payloads) != len(batch):
                raise EnvironmentError(
                    f"Hourly ERA5 returned {len(payloads)} locations for {len(batch)} requests."
                )
            rows.extend(
                _hourly_window_features(point, payload)
                for (_, point), payload in zip(batch.iterrows(), payloads)
            )
    return pd.DataFrame(rows)


def load_bylot_lemming_index(
    catalog_root: Path | str = DEFAULT_CATALOG_ROOT,
    force: bool = False,
) -> pd.DataFrame:
    root = Path(catalog_root)
    target = root / "predictors" / "bylot-lemmings.parquet"
    if target.exists() and not force:
        return pd.read_parquet(target)
    response = _get(BYLOT_ZENODO_URL, timeout=180)
    with ZipFile(BytesIO(response.content)) as archive:
        matches = [
            name for name in archive.namelist()
            if name.endswith("lemming_density_1993-2019.csv")
        ]
        if len(matches) != 1:
            raise EnvironmentError("The Bylot package did not contain the expected lemming series.")
        raw = pd.read_csv(archive.open(matches[0]))
    wet = pd.to_numeric(raw["Both Wet habitat"], errors="coerce")
    mesic = pd.to_numeric(raw["Both Mesic habitat"], errors="coerce")
    output = pd.DataFrame(
        {
            "year": pd.to_numeric(raw.Year, errors="raise").astype(int),
            "lemming_density_index": pd.concat([wet, mesic], axis=1).mean(axis=1),
            "lemming_source": f"Bylot abundance package; doi:{BYLOT_SOURCE_DOI}",
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.parquet")
    output.to_parquet(temporary, index=False)
    temporary.replace(target)
    return output


def join_bylot_lemmings(points: pd.DataFrame, lemmings: pd.DataFrame) -> pd.DataFrame:
    fox = points[points.species.eq("Vulpes lagopus")][JOIN_KEYS + ["year"]].copy()
    if fox.empty:
        return pd.DataFrame(columns=JOIN_KEYS + ["lemming_density_index", "lemming_source"])
    return fox.merge(lemmings, on="year", how="left").drop(columns="year")


@dataclass(frozen=True)
class ErddapDataset:
    dataset_id: str
    variable: str
    output: str
    dimensions: tuple[str, ...]
    minimum_year: int | None = None
    maximum_year: int | None = None


MARINE_DATASETS = (
    ErddapDataset(
        "erdSW2018chlamday", "chlorophyll", "chlorophyll_a",
        ("time", "latitude", "longitude"), 1997, 2010,
    ),
    ErddapDataset(
        "erdMH1ppmday", "productivity", "primary_productivity",
        ("time", "altitude", "latitude", "longitude"), 2003, None,
    ),
    ErddapDataset(
        "srtm30plus_LonPM180", "z", "bathymetry_m", ("latitude", "longitude"),
    ),
    ErddapDataset(
        "FRD_SSTgradsmo", "SSTgrad", "sst_front_gradient",
        ("time", "latitude", "longitude"),
    ),
)


def _constraint(value) -> str:
    return f"[({quote(str(value), safe='-:.TZ')})]"


def _erddap_point(dataset: ErddapDataset, point: pd.Series) -> float:
    year = int(point.year)
    if dataset.minimum_year and year < dataset.minimum_year:
        return float("nan")
    if dataset.maximum_year and year > dataset.maximum_year:
        return float("nan")
    values = {
        "time": f"{pd.Timestamp(point.anchor_date).date()}T00:00:00Z",
        "altitude": 0,
        "latitude": round(float(point.latitude), 2),
        "longitude": round(float(point.longitude), 2),
    }
    query = dataset.variable + "".join(_constraint(values[name]) for name in dataset.dimensions)
    url = f"{ERDDAP}/{dataset.dataset_id}.csv?{query}"
    try:
        response = _get(url, timeout=120)
        frame = pd.read_csv(StringIO(response.text), skiprows=[1])
        return float(pd.to_numeric(frame[dataset.variable], errors="coerce").iloc[0])
    except (EnvironmentError, KeyError, IndexError, TypeError, ValueError, pd.errors.ParserError):
        return float("nan")


def fetch_erddap_marine_features(points: pd.DataFrame) -> pd.DataFrame:
    marine = points[points.species.eq("Balaenoptera musculus")].copy()
    if marine.empty:
        return pd.DataFrame(columns=JOIN_KEYS + [item.output for item in MARINE_DATASETS])
    requests_to_make: dict[tuple, tuple[ErddapDataset, pd.Series]] = {}
    for _, point in marine.iterrows():
        for dataset in MARINE_DATASETS:
            time_key = (
                str(pd.Timestamp(point.anchor_date).to_period("M"))
                if "time" in dataset.dimensions else "static"
            )
            cache_key = (
                dataset.dataset_id, time_key,
                round(float(point.latitude), 1), round(float(point.longitude), 1),
            )
            cached_point = point.copy()
            cached_point.latitude = cache_key[2]
            cached_point.longitude = cache_key[3]
            requests_to_make.setdefault(cache_key, (dataset, cached_point))

    def fetch(item):
        key, (dataset, cached_point) = item
        return key, _erddap_point(dataset, cached_point)

    # ERDDAP point reads are network-bound. A modestly wider pool keeps a full
    # historical refresh practical while the shared request helper still backs
    # off on 429 and 5xx responses.
    with ThreadPoolExecutor(max_workers=16) as pool:
        cache = dict(pool.map(fetch, requests_to_make.items()))
    rows = []
    for _, point in marine.iterrows():
        row = {key: point[key] for key in JOIN_KEYS}
        for dataset in MARINE_DATASETS:
            time_key = (
                str(pd.Timestamp(point.anchor_date).to_period("M"))
                if "time" in dataset.dimensions else "static"
            )
            cache_key = (
                dataset.dataset_id,
                time_key,
                round(float(point.latitude), 1),
                round(float(point.longitude), 1),
            )
            row[dataset.output] = cache[cache_key]
        row["marine_biology_source"] = "NOAA CoastWatch ERDDAP"
        rows.append(row)
    return pd.DataFrame(rows)


class NsidcSeaIceClient:
    GRID_DATASET = "nsidcCDRice_nh_grid"
    ICE_DATASET = "nsidcG02202v6nh1day"

    def __init__(self):
        self._grid: pd.DataFrame | None = None
        self._nearest: dict[tuple[float, float], tuple[float, float]] = {}
        self._values: dict[tuple[str, float, float], tuple[float, float, float]] = {}

    def _load_grid(self) -> pd.DataFrame:
        if self._grid is None:
            url = (
                f"{ERDDAP}/{self.GRID_DATASET}.csv?"
                "latitude%5B0:1:447%5D%5B0:1:303%5D,"
                "longitude%5B0:1:447%5D%5B0:1:303%5D"
            )
            self._grid = pd.read_csv(StringIO(_get(url, timeout=180).text), skiprows=[1])
        return self._grid

    def nearest_xy(self, latitude: float, longitude: float) -> tuple[float, float]:
        key = (round(float(latitude), 2), round(float(longitude), 2))
        if key not in self._nearest:
            grid = self._load_grid()
            lat = np.radians(grid.latitude.to_numpy(float))
            lon = np.radians(grid.longitude.to_numpy(float))
            target_lat, target_lon = np.radians(key)
            score = (
                np.square(lat - target_lat)
                + np.square(np.cos(target_lat) * (lon - target_lon))
            )
            nearest = grid.iloc[int(np.nanargmin(score))]
            self._nearest[key] = (float(nearest.ygrid), float(nearest.xgrid))
        return self._nearest[key]

    def sample(self, date, latitude: float, longitude: float) -> tuple[float, float, float]:
        y, x = self.nearest_xy(latitude, longitude)
        key = (str(pd.Timestamp(date).date()), y, x)
        if key not in self._values:
            time_value = f"{key[0]}T00:00:00Z"
            variables = ["cdr_seaice_conc", "cdr_seaice_conc_stdev"]
            y_index = int(round((5_837_500.0 - y) / 25_000.0))
            x_index = int(round((x + 3_837_500.0) / 25_000.0))
            radius = 6
            y_start, y_stop = max(0, y_index - radius), min(447, y_index + radius)
            x_start, x_stop = max(0, x_index - radius), min(303, x_index + radius)
            queries = [
                variable + _constraint(time_value)
                + f"[{y_start}:1:{y_stop}][{x_start}:1:{x_stop}]"
                for variable in variables
            ]
            url = f"{ERDDAP}/{self.ICE_DATASET}.csv?" + ",".join(queries)
            try:
                frame = pd.read_csv(StringIO(_get(url, timeout=120).text), skiprows=[1])
                concentration = pd.to_numeric(frame[variables[0]], errors="coerce")
                valid = concentration.between(0, 1)
                candidates = frame[valid].copy()
                candidates[variables[0]] = concentration[valid]
                if candidates.empty:
                    values = (float("nan"), float("nan"), float("nan"))
                else:
                    candidates["distance"] = np.hypot(
                        pd.to_numeric(candidates.y, errors="coerce") - y,
                        pd.to_numeric(candidates.x, errors="coerce") - x,
                    ) / 1000.0
                    nearest = candidates.sort_values("distance").iloc[0]
                    values = (
                        float(nearest[variables[0]]),
                        float(pd.to_numeric(pd.Series([nearest[variables[1]]]), errors="coerce").iloc[0]),
                        float(nearest.distance),
                    )
            except (EnvironmentError, KeyError, IndexError, ValueError, pd.errors.ParserError):
                values = (float("nan"), float("nan"), float("nan"))
            self._values[key] = values
        return self._values[key]


def fetch_fox_sea_ice_features(points: pd.DataFrame) -> pd.DataFrame:
    fox = points[points.species.eq("Vulpes lagopus")].copy()
    if fox.empty:
        return pd.DataFrame(columns=JOIN_KEYS + ["sea_ice_concentration"])
    client = NsidcSeaIceClient()
    client._load_grid()
    unique: dict[tuple, tuple] = {}
    for _, point in fox.iterrows():
        y, x = client.nearest_xy(float(point.latitude), float(point.longitude))
        key = (str(pd.Timestamp(point.anchor_date).date()), y, x)
        unique.setdefault(key, (point.anchor_date, float(point.latitude), float(point.longitude)))

    def sample(item):
        key, arguments = item
        return key, client.sample(*arguments)

    with ThreadPoolExecutor(max_workers=8) as pool:
        sampled = dict(pool.map(sample, unique.items()))
    rows = []
    for _, point in fox.iterrows():
        y, x = client.nearest_xy(float(point.latitude), float(point.longitude))
        key = (str(pd.Timestamp(point.anchor_date).date()), y, x)
        concentration, stdev, distance = sampled[key]
        rows.append(
            {
                **{key: point[key] for key in JOIN_KEYS},
                "sea_ice_concentration": concentration,
                "sea_ice_concentration_stdev": stdev,
                "distance_to_sea_ice_km": distance,
                "sea_ice_source": "NOAA/NSIDC CDR v6 via NOAA CoastWatch ERDDAP",
            }
        )
    return pd.DataFrame(rows)


def fetch_glorys_journey_features(
    points: pd.DataFrame,
    dataset_id: str = "cmems_mod_glo_phy_my_0.083deg_P1D-m",
) -> pd.DataFrame:
    marine = points[points.species.eq("Balaenoptera musculus")].copy()
    if marine.empty:
        return pd.DataFrame(columns=JOIN_KEYS)
    try:
        import copernicusmarine
    except ImportError as exc:
        raise EnvironmentError(
            "GLORYS requires requirements-marine.txt and `copernicusmarine login`."
        ) from exc
    username = os.getenv("COPERNICUSMARINE_SERVICE_USERNAME") or os.getenv(
        "COPERNICUS_USERNAME"
    )
    password = os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD") or os.getenv(
        "COPERNICUS_PASSWORD"
    )
    credentials_file = Path.home() / ".copernicusmarine" / ".copernicusmarine-credentials"
    if not (username and password) and not credentials_file.exists():
        raise EnvironmentError(
            "GLORYS needs Copernicus Marine credentials. Set "
            "COPERNICUSMARINE_SERVICE_USERNAME and "
            "COPERNICUSMARINE_SERVICE_PASSWORD, or run `copernicusmarine login`."
        )
    rows = []
    for year, group in marine.groupby("year", sort=True):
        start = pd.to_datetime(group.start_date).min()
        end = pd.to_datetime(group.end_date).max() + pd.Timedelta(days=1)
        try:
            cube = copernicusmarine.open_dataset(
                dataset_id=dataset_id,
                username=username,
                password=password,
                variables=["uo", "vo", "thetao", "so", "zos", "mlotst"],
                minimum_longitude=float(group.longitude.min()) - 0.2,
                maximum_longitude=float(group.longitude.max()) + 0.2,
                minimum_latitude=float(group.latitude.min()) - 0.2,
                maximum_latitude=float(group.latitude.max()) + 0.2,
                start_datetime=start.isoformat(),
                end_datetime=end.isoformat(),
            )
        except Exception as exc:
            raise EnvironmentError(f"GLORYS year {year} could not be opened: {exc}") from exc
        if cube is None:
            raise EnvironmentError(
                f"GLORYS year {year} returned no data; verify Copernicus Marine credentials."
            )
        if "depth" in cube.coords:
            cube = cube.sel(depth=0, method="nearest")
        for _, point in group.iterrows():
            try:
                selected = cube.sel(
                    time=pd.Timestamp(point.anchor_date),
                    latitude=float(point.latitude),
                    longitude=float(point.longitude),
                    method="nearest",
                )
                u = float(selected.uo.values) * 3.6
                v = float(selected.vo.values) * 3.6
                bearing = math.radians(float(point.route_bearing_degrees))
                rows.append(
                    {
                        **{key: point[key] for key in JOIN_KEYS},
                        "sea_surface_temperature": float(selected.thetao.values),
                        "marine_current_u": u,
                        "marine_current_v": v,
                        "marine_current_tailwind": u * math.sin(bearing) + v * math.cos(bearing),
                        "mixed_layer_depth": float(selected.mlotst.values),
                        "sea_surface_height": float(selected.zos.values),
                        "sea_surface_salinity": float(selected.so.values),
                        "glorys_source": "Copernicus Marine GLORYS12V1",
                    }
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
    return pd.DataFrame(rows)


def merge_feature_frames(base: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
    output = base.copy()
    for features in frames:
        if features is None or features.empty:
            continue
        if features.duplicated(JOIN_KEYS).any():
            raise EnvironmentError("An enrichment source returned duplicate journey-window rows.")
        additions = [column for column in features if column not in JOIN_KEYS]
        output = output.merge(features, on=JOIN_KEYS, how="left", suffixes=("", "__new"))
        for column in additions:
            replacement = f"{column}__new"
            if replacement in output:
                if column in output:
                    output[column] = output[replacement].where(
                        output[replacement].notna(), output[column]
                    )
                else:
                    output[column] = output[replacement]
                output.drop(columns=replacement, inplace=True)
    return output
