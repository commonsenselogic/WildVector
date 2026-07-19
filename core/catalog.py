from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable

import duckdb
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "catalog" / "studies.json"
DEFAULT_CATALOG_ROOT = PROJECT_ROOT / "data" / "catalog"
REPOSITORY_API = "https://datarepository.movebank.org/server/api"
OPEN_LICENSES = {
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "https://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "https://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
}

RAW_ALIASES = {
    "individual-local-identifier": "animal_id",
    "individual_local_identifier": "animal_id",
    "individual-taxon-canonical-name": "species",
    "individual_taxon_canonical_name": "species",
    "timestamp": "timestamp_utc",
    "location-lat": "latitude",
    "location_lat": "latitude",
    "location-long": "longitude",
    "location_long": "longitude",
    "sensor-type": "sensor_type",
    "sensor_type": "sensor_type",
    "visible": "visible",
}
RAW_COLUMNS = set(RAW_ALIASES)
TELEMETRY_COLUMNS = [
    "study_id",
    "study_key",
    "population",
    "population_key",
    "species",
    "species_key",
    "animal_id",
    "timestamp_utc",
    "latitude",
    "longitude",
    "sensor_type",
    "movement_type",
    "hemisphere",
    "season",
    "year",
    "source_doi",
    "license",
]


class CatalogError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogStudy:
    key: str
    item_id: str
    study_id: str
    title: str
    population: str
    taxa: tuple[str, ...]
    movement_type: str
    hemisphere: str
    animals: int
    locations: int
    doi: str
    license: str

    @classmethod
    def from_dict(cls, value: dict) -> "CatalogStudy":
        return cls(**{**value, "taxa": tuple(value["taxa"])})


@dataclass(frozen=True)
class RefreshResult:
    study_key: str
    status: str
    rows: int
    animals: int
    species: int
    years: int
    files: int
    bytes_downloaded: int
    message: str = ""


def slug(value: object) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")
    return clean or "unknown"


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> list[CatalogStudy]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    studies = [CatalogStudy.from_dict(item) for item in payload.get("studies", [])]
    if not 25 <= len(studies) <= 50:
        raise CatalogError(f"The selected manifest must contain 25-50 studies; found {len(studies)}.")
    keys = [study.key for study in studies]
    if len(keys) != len(set(keys)):
        raise CatalogError("Study keys must be unique.")
    for study in studies:
        if study.license not in set(OPEN_LICENSES.values()):
            raise CatalogError(f"{study.key} is not configured with an approved open license.")
        if study.movement_type not in {"terrestrial", "aerial", "marine"}:
            raise CatalogError(f"{study.key} has an unsupported movement type.")
    return studies


def season_for_month(month: int, hemisphere: str) -> str:
    north = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring migration",
        4: "spring migration",
        5: "spring migration",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "fall migration",
        10: "fall migration",
        11: "fall migration",
    }
    if hemisphere == "north":
        return north[int(month)]
    reverse = {
        "winter": "summer",
        "spring migration": "fall migration",
        "summer": "winter",
        "fall migration": "spring migration",
    }
    return reverse[north[int(month)]]


def _metadata_values(item: dict, field: str) -> list[str]:
    return [str(value["value"]) for value in item.get("metadata", {}).get(field, [])]


class MovebankRepositoryClient:
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WildVector catalog builder/1.0"})

    def _json(self, url: str) -> dict:
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise CatalogError(f"Movebank repository request failed: {exc}") from exc

    def item(self, item_id: str) -> dict:
        return self._json(f"{REPOSITORY_API}/core/items/{item_id}")

    def validate(self, study: CatalogStudy) -> dict:
        item = self.item(study.item_id)
        license_uris = _metadata_values(item, "dc.rights.uri")
        license_uri = license_uris[0] if license_uris else ""
        normalized_license = OPEN_LICENSES.get(license_uri)
        if normalized_license is None:
            raise CatalogError(
                f"{study.key} is no longer offered under CC0 or CC BY 4.0 ({license_uri or 'missing'})."
            )
        if normalized_license != study.license:
            raise CatalogError(
                f"{study.key} license changed from {study.license} to {normalized_license}."
            )
        repository_study_ids = _metadata_values(item, "mdr.study.id")
        if repository_study_ids and study.study_id not in repository_study_ids:
            raise CatalogError(f"{study.key} Movebank study ID no longer matches repository metadata.")
        repository_taxa = {value.casefold() for value in _metadata_values(item, "dwc.ScientificName")}
        expected_taxa = {value.casefold() for value in study.taxa}
        if repository_taxa and not repository_taxa.intersection(expected_taxa):
            raise CatalogError(f"{study.key} taxon metadata no longer matches the manifest.")
        return item

    def csv_bitstreams(self, study: CatalogStudy) -> list[dict]:
        bundles = self._json(f"{REPOSITORY_API}/core/items/{study.item_id}/bundles?size=100")
        originals = [
            bundle
            for bundle in bundles.get("_embedded", {}).get("bundles", [])
            if bundle.get("name") == "ORIGINAL"
        ]
        if not originals:
            raise CatalogError(f"{study.key} has no ORIGINAL file bundle.")
        url = originals[0]["_links"]["bitstreams"]["href"] + "?size=100"
        payload = self._json(url)
        candidates = []
        for bitstream in payload.get("_embedded", {}).get("bitstreams", []):
            name = str(bitstream.get("name", ""))
            lower = name.casefold()
            if not lower.endswith(".csv") or "reference-data" in lower:
                continue
            candidates.append(
                {
                    "name": name,
                    "size": int(bitstream.get("sizeBytes", 0) or 0),
                    "url": bitstream["_links"]["content"]["href"],
                }
            )
        if not candidates:
            raise CatalogError(f"{study.key} has no telemetry CSV in its ORIGINAL bundle.")
        return candidates

    def download(self, bitstream: dict, destination: Path, maximum_bytes: int) -> int:
        expected = int(bitstream.get("size", 0) or 0)
        if expected and expected > maximum_bytes:
            raise CatalogError(
                f"{bitstream['name']} is {expected / 1_000_000:.1f} MB, above the configured "
                f"{maximum_bytes / 1_000_000:.0f} MB per-file limit."
            )
        try:
            with self.session.get(bitstream["url"], stream=True, timeout=self.timeout) as response:
                response.raise_for_status()
                total = 0
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > maximum_bytes:
                            raise CatalogError(f"{bitstream['name']} exceeded the download limit.")
                        handle.write(chunk)
                return total
        except requests.RequestException as exc:
            raise CatalogError(f"Could not download {bitstream['name']}: {exc}") from exc


def normalize_repository_chunk(raw: pd.DataFrame, study: CatalogStudy) -> pd.DataFrame:
    out = raw.rename(columns=RAW_ALIASES).copy()
    required = {"animal_id", "timestamp_utc", "latitude", "longitude"}
    if not required.issubset(out):
        return pd.DataFrame(columns=TELEMETRY_COLUMNS)
    if "species" not in out:
        if len(study.taxa) != 1:
            raise CatalogError(f"{study.key} is multi-species but its CSV has no taxon column.")
        out["species"] = study.taxa[0]
    if "sensor_type" not in out:
        out["sensor_type"] = "unknown"
    out["timestamp_utc"] = pd.to_datetime(out.timestamp_utc, utc=True, errors="coerce")
    out["latitude"] = pd.to_numeric(out.latitude, errors="coerce")
    out["longitude"] = pd.to_numeric(out.longitude, errors="coerce")
    if "visible" in out:
        visible = out.visible.astype(str).str.casefold()
        out = out[~visible.isin({"false", "0", "no"})]
    out = out[
        out.animal_id.notna()
        & out.timestamp_utc.notna()
        & out.latitude.between(-90, 90)
        & out.longitude.between(-180, 180)
    ].copy()
    expected = {value.casefold() for value in study.taxa}
    observed = out.species.astype(str).str.strip()
    keep = observed.str.casefold().isin(expected)
    out = out[keep].copy()
    if out.empty:
        return pd.DataFrame(columns=TELEMETRY_COLUMNS)
    out["species"] = observed[keep].values
    out["animal_id"] = study.key + ":" + out.animal_id.astype(str)
    out["study_id"] = study.study_id
    out["study_key"] = study.key
    out["population"] = study.population
    out["population_key"] = slug(study.population)
    out["species_key"] = out.species.map(slug)
    out["movement_type"] = study.movement_type
    out["hemisphere"] = study.hemisphere
    out["season"] = out.timestamp_utc.dt.month.map(
        lambda month: season_for_month(month, study.hemisphere)
    )
    out["year"] = out.timestamp_utc.dt.year.astype("int32")
    out["source_doi"] = study.doi
    out["license"] = study.license
    return out[TELEMETRY_COLUMNS].drop_duplicates(
        ["animal_id", "timestamp_utc", "latitude", "longitude"]
    )


def _partition_path(base: Path, frame: pd.DataFrame) -> Path:
    row = frame.iloc[0]
    return (
        base
        / f"species_key={row.species_key}"
        / f"population_key={row.population_key}"
        / f"season_key={slug(row.season)}"
        / f"year={int(row.year)}"
    )


def _write_frame_partitions(frame: pd.DataFrame, base: Path, counter: int) -> tuple[int, int]:
    files = 0
    rows = 0
    partition_columns = ["species_key", "population_key", "season", "year"]
    for _, group in frame.groupby(partition_columns, sort=True, dropna=False):
        destination = _partition_path(base, group)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"part-{counter:06d}-{files:03d}.parquet"
        group.to_parquet(path, index=False, compression="zstd")
        files += 1
        rows += len(group)
    return files, rows


def _safe_replace_directory(staging: Path, target: Path, root: Path) -> None:
    root = root.resolve()
    target = target.resolve()
    staging = staging.resolve()
    if root not in target.parents or root not in staging.parents:
        raise CatalogError("Catalog replacement target escaped the configured catalog root.")
    backup = root / f".backup-{target.name}"
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        target.rename(backup)
    try:
        staging.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def refresh_study(
    study: CatalogStudy,
    catalog_root: Path | str = DEFAULT_CATALOG_ROOT,
    maximum_file_bytes: int = 750_000_000,
    chunk_rows: int = 100_000,
    client: MovebankRepositoryClient | None = None,
) -> RefreshResult:
    root = Path(catalog_root)
    telemetry_root = root / "telemetry"
    telemetry_root.mkdir(parents=True, exist_ok=True)
    target = telemetry_root / f"study_id={study.study_id}"
    staging = telemetry_root / f".staging-study_id={study.study_id}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    client = client or MovebankRepositoryClient()
    client.validate(study)
    bitstreams = client.csv_bitstreams(study)
    rows = files = downloaded = 0
    animal_ids: set[str] = set()
    species: set[str] = set()
    years: set[int] = set()
    try:
        with tempfile.TemporaryDirectory(prefix=f"wildvector-{study.key}-") as temp:
            temp_root = Path(temp)
            for source_index, bitstream in enumerate(bitstreams):
                source = temp_root / f"{source_index:03d}.csv"
                downloaded += client.download(bitstream, source, maximum_file_bytes)
                try:
                    chunks = pd.read_csv(
                        source,
                        usecols=lambda name: name in RAW_COLUMNS,
                        chunksize=chunk_rows,
                        low_memory=False,
                    )
                    for chunk_index, chunk in enumerate(chunks):
                        normalized = normalize_repository_chunk(chunk, study)
                        if normalized.empty:
                            continue
                        animal_ids.update(normalized.animal_id.astype(str).unique())
                        species.update(normalized.species.astype(str).unique())
                        years.update(int(value) for value in normalized.year.unique())
                        written_files, written_rows = _write_frame_partitions(
                            normalized, staging, source_index * 1_000_000 + chunk_index
                        )
                        files += written_files
                        rows += written_rows
                except (ValueError, pd.errors.ParserError) as exc:
                    raise CatalogError(f"Could not parse {bitstream['name']}: {exc}") from exc
        if rows == 0:
            raise CatalogError(f"{study.key} produced no valid telemetry rows.")
        _safe_replace_directory(staging, target, telemetry_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return RefreshResult(
        study.key,
        "ready",
        rows,
        len(animal_ids),
        len(species),
        len(years),
        files,
        downloaded,
    )


def _study_table(studies: Iterable[CatalogStudy]) -> pd.DataFrame:
    rows = []
    for study in studies:
        item = asdict(study)
        item["taxa"] = "; ".join(study.taxa)
        rows.append(item)
    return pd.DataFrame(rows)


def rebuild_database(
    catalog_root: Path | str = DEFAULT_CATALOG_ROOT,
    manifest_path: Path | str = DEFAULT_MANIFEST,
) -> Path:
    root = Path(catalog_root)
    root.mkdir(parents=True, exist_ok=True)
    database_path = root / "wildvector.duckdb"
    studies = load_manifest(manifest_path)
    telemetry_files = list((root / "telemetry").glob("**/*.parquet"))
    with duckdb.connect(str(database_path)) as connection:
        connection.register("manifest_frame", _study_table(studies))
        connection.execute("CREATE OR REPLACE TABLE studies AS SELECT * FROM manifest_frame")
        connection.unregister("manifest_frame")
        existing = connection.execute(
            "SELECT table_type FROM information_schema.tables WHERE table_name = 'telemetry'"
        ).fetchone()
        if existing:
            connection.execute(
                "DROP VIEW telemetry" if existing[0] == "VIEW" else "DROP TABLE telemetry"
            )
        if telemetry_files:
            parquet_glob = str((root / "telemetry" / "**" / "*.parquet").resolve()).replace("\\", "/")
            connection.execute(
                f"CREATE VIEW telemetry AS "
                f"SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning=true, union_by_name=true)"
            )
        else:
            connection.execute(
                "CREATE TABLE telemetry ("
                "study_id VARCHAR, study_key VARCHAR, population VARCHAR, population_key VARCHAR, "
                "species VARCHAR, species_key VARCHAR, animal_id VARCHAR, timestamp_utc TIMESTAMPTZ, "
                "latitude DOUBLE, longitude DOUBLE, sensor_type VARCHAR, movement_type VARCHAR, "
                "hemisphere VARCHAR, season VARCHAR, year INTEGER, source_doi VARCHAR, license VARCHAR)"
            )
        connection.execute(
            "CREATE OR REPLACE TABLE catalog_build AS SELECT ? AS refreshed_at, ? AS manifest_studies, "
            "(SELECT count(*) FROM telemetry) AS telemetry_rows",
            [datetime.now(timezone.utc), len(studies)],
        )
        for view_name, file_name in {
            "corridors": "corridors.parquet",
            "journeys": "journeys.parquet",
            "representative_journeys": "representative-journeys.parquet",
            "environment": "environment.parquet",
            "journey_environment": "environment-journeys.parquet",
            "external_validation": "external-validation.parquet",
        }.items():
            path = (root / file_name).resolve()
            if path.exists():
                normalized = str(path).replace("\\", "/")
                connection.execute(
                    f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{normalized}')"
                )
    return database_path


class CatalogStore:
    def __init__(self, database_path: Path | str = DEFAULT_CATALOG_ROOT / "wildvector.duckdb"):
        self.database_path = Path(database_path)

    @property
    def available(self) -> bool:
        return self.database_path.exists()

    def query(self, sql: str, parameters: list | tuple | None = None) -> pd.DataFrame:
        if not self.available:
            raise CatalogError(
                "The local catalog has not been built. Run scripts/refresh_catalog.py first."
            )
        with duckdb.connect(str(self.database_path), read_only=True) as connection:
            return connection.execute(sql, parameters or []).fetch_df()

    def _relation_available(self, name: str) -> bool:
        return not self.query(
            """
            SELECT table_name FROM information_schema.tables WHERE table_name = ?
            UNION ALL
            SELECT table_name FROM information_schema.views WHERE table_name = ?
            """,
            [name, name],
        ).empty

    def populations(self) -> pd.DataFrame:
        return self.query(
            """
            SELECT species, population, movement_type, season,
                   count(DISTINCT year) AS years,
                   count(DISTINCT animal_id) AS animals,
                   count(*) AS locations,
                   min(timestamp_utc) AS first_seen,
                   max(timestamp_utc) AS last_seen
            FROM telemetry
            GROUP BY species, population, movement_type, season
            HAVING count(*) >= 2
            ORDER BY species, population, season
            """
        )

    def telemetry(
        self,
        species: str,
        population: str,
        season: str,
        max_journeys: int = 24,
    ) -> pd.DataFrame:
        return self.query(
            """
            WITH selected AS (
              SELECT animal_id, year
              FROM telemetry
              WHERE species = ? AND population = ? AND season = ?
              GROUP BY animal_id, year
              ORDER BY count(*) DESC
              LIMIT ?
            )
            SELECT t.*
            FROM telemetry t
            JOIN selected s USING (animal_id, year)
            WHERE t.species = ? AND t.population = ? AND t.season = ?
            ORDER BY t.animal_id, t.year, t.timestamp_utc
            """,
            [species, population, season, max_journeys, species, population, season],
        )

    @staticmethod
    def _migration_journeys_sql() -> str:
        # Net displacement excludes local foraging loops and seasonal home-range movement.
        return """
            WITH endpoints AS (
              SELECT species, population, movement_type, animal_id, year, season,
                     count(*) AS points,
                     arg_min(latitude, timestamp_utc) AS start_latitude,
                     arg_min(longitude, timestamp_utc) AS start_longitude,
                     arg_max(latitude, timestamp_utc) AS end_latitude,
                     arg_max(longitude, timestamp_utc) AS end_longitude
              FROM telemetry
              WHERE season IN ('spring migration', 'fall migration')
              GROUP BY species, population, movement_type, animal_id, year, season
              HAVING count(*) >= 4
            ), distances AS (
              SELECT *, 6371.0088 * 2 * asin(sqrt(least(1.0,
                       pow(sin(radians(end_latitude - start_latitude) / 2), 2) +
                       cos(radians(start_latitude)) * cos(radians(end_latitude)) *
                       pow(sin(radians(end_longitude - start_longitude) / 2), 2)
                     ))) AS net_distance_km
              FROM endpoints
            )
            SELECT * FROM distances
            WHERE net_distance_km >= CASE movement_type
              WHEN 'aerial' THEN 300.0
              WHEN 'marine' THEN 200.0
              ELSE 50.0 END
        """

    def migration_populations(self, minimum_journeys_per_season: int = 3) -> pd.DataFrame:
        if self._relation_available("journeys"):
            return self.query(
                """
                WITH qualifying AS (
                  SELECT * FROM journeys
                  WHERE season IN ('spring migration', 'fall migration')
                    AND points >= 4
                    AND net_distance_km >= CASE movement_type
                      WHEN 'aerial' THEN 300.0
                      WHEN 'marine' THEN 200.0
                      ELSE 50.0 END
                )
                SELECT species, population, movement_type,
                       count(*) AS journeys,
                       count(DISTINCT animal_id) AS animals,
                       count(DISTINCT year) AS years,
                       min(net_distance_km) AS shortest_journey_km,
                       median(net_distance_km) AS typical_journey_km,
                       max(net_distance_km) AS longest_journey_km
                FROM qualifying
                GROUP BY species, population, movement_type
                HAVING count(*) FILTER (WHERE season = 'spring migration') >= ?
                   AND count(*) FILTER (WHERE season = 'fall migration') >= ?
                ORDER BY species, population
                """,
                [minimum_journeys_per_season, minimum_journeys_per_season],
            )
        qualifying = self._migration_journeys_sql()
        return self.query(
            f"""
            WITH qualifying AS ({qualifying})
            SELECT species, population, movement_type,
                   count(*) AS journeys,
                   count(DISTINCT animal_id) AS animals,
                   count(DISTINCT year) AS years,
                   min(net_distance_km) AS shortest_journey_km,
                   median(net_distance_km) AS typical_journey_km,
                   max(net_distance_km) AS longest_journey_km
            FROM qualifying
            GROUP BY species, population, movement_type
            HAVING count(*) FILTER (WHERE season = 'spring migration') >= ?
               AND count(*) FILTER (WHERE season = 'fall migration') >= ?
            ORDER BY species, population
            """,
            [minimum_journeys_per_season, minimum_journeys_per_season],
        )

    def migration_telemetry(
        self,
        species: str,
        population: str,
        max_journeys: int = 36,
    ) -> pd.DataFrame:
        if self._relation_available("journeys"):
            return self.query(
                """
                WITH ranked AS (
                  SELECT *, row_number() OVER (
                    PARTITION BY season ORDER BY net_distance_km DESC, animal_id, year
                  ) AS season_rank
                  FROM journeys
                  WHERE species = ? AND population = ?
                    AND season IN ('spring migration', 'fall migration')
                    AND points >= 4
                    AND net_distance_km >= CASE movement_type
                      WHEN 'aerial' THEN 300.0
                      WHEN 'marine' THEN 200.0
                      ELSE 50.0 END
                ), selected AS (
                  SELECT * FROM ranked
                  WHERE season_rank <= greatest(1, ceil(? / 2.0))
                  ORDER BY net_distance_km DESC
                  LIMIT ?
                )
                SELECT t.*
                FROM telemetry t
                JOIN selected s
                  ON t.species = s.species AND t.population = s.population
                 AND t.animal_id = s.animal_id AND t.year = s.year AND t.season = s.season
                ORDER BY t.season, t.animal_id, t.year, t.timestamp_utc
                """,
                [species, population, max_journeys, max_journeys],
            )
        qualifying = self._migration_journeys_sql()
        return self.query(
            f"""
            WITH qualifying AS ({qualifying}), ranked AS (
              SELECT *, row_number() OVER (
                PARTITION BY season ORDER BY net_distance_km DESC, animal_id, year
              ) AS season_rank
              FROM qualifying
              WHERE species = ? AND population = ?
            ), selected AS (
              SELECT * FROM ranked
              WHERE season_rank <= greatest(1, ceil(? / 2.0))
              ORDER BY net_distance_km DESC
              LIMIT ?
            )
            SELECT t.*
            FROM telemetry t
            JOIN selected s
              ON t.species = s.species AND t.population = s.population
             AND t.animal_id = s.animal_id AND t.year = s.year AND t.season = s.season
            ORDER BY t.season, t.animal_id, t.year, t.timestamp_utc
            """,
            [species, population, max_journeys, max_journeys],
        )


def write_refresh_report(
    results: Iterable[RefreshResult],
    catalog_root: Path | str = DEFAULT_CATALOG_ROOT,
) -> Path:
    root = Path(catalog_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "refresh-report.json"
    payload = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "results": [asdict(result) for result in results],
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path
