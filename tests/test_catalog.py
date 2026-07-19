from pathlib import Path

import pandas as pd

from core.catalog import (
    CatalogStore,
    load_manifest,
    normalize_repository_chunk,
    rebuild_database,
    season_for_month,
)


def test_manifest_selects_30_open_balanced_studies():
    studies = load_manifest()
    assert len(studies) == 30
    counts = pd.Series([study.movement_type for study in studies]).value_counts()
    assert counts.to_dict() == {"aerial": 12, "terrestrial": 9, "marine": 9}
    assert all(study.license in {"CC0-1.0", "CC-BY-4.0"} for study in studies)


def test_repository_normalization_adds_population_season_and_namespaces():
    study = next(study for study in load_manifest() if study.key == "whooping_crane_migration")
    raw = pd.DataFrame(
        {
            "individual-local-identifier": ["crane-1", "crane-1"],
            "individual-taxon-canonical-name": ["Grus americana", "Grus americana"],
            "timestamp": ["2017-04-01T00:00:00Z", "2017-10-01T00:00:00Z"],
            "location-lat": [30.0, 45.0],
            "location-long": [-90.0, -85.0],
            "sensor-type": ["GPS", "GPS"],
        }
    )
    normalized = normalize_repository_chunk(raw, study)
    assert normalized.animal_id.str.startswith("whooping_crane_migration:").all()
    assert set(normalized.season) == {"spring migration", "fall migration"}
    assert normalized.population.nunique() == 1


def test_seasons_reverse_in_southern_hemisphere():
    assert season_for_month(4, "north") == "spring migration"
    assert season_for_month(4, "south") == "fall migration"
    assert season_for_month(7, "south") == "winter"


def test_duckdb_queries_partitioned_local_catalog(tmp_path: Path):
    telemetry_root = (
        tmp_path
        / "telemetry"
        / "study_id=1"
        / "species_key=test-bird"
        / "population_key=test-flyway"
        / "season_key=spring-migration"
        / "year=2020"
    )
    telemetry_root.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "study_id": ["1", "1"],
            "study_key": ["test", "test"],
            "population": ["Test Flyway", "Test Flyway"],
            "population_key": ["test-flyway", "test-flyway"],
            "species": ["Test bird", "Test bird"],
            "species_key": ["test-bird", "test-bird"],
            "animal_id": ["a", "a"],
            "timestamp_utc": pd.to_datetime(
                ["2020-04-01T00:00:00Z", "2020-04-02T00:00:00Z"]
            ),
            "latitude": [1.0, 2.0],
            "longitude": [3.0, 4.0],
            "sensor_type": ["GPS", "GPS"],
            "movement_type": ["aerial", "aerial"],
            "hemisphere": ["north", "north"],
            "season": ["spring migration", "spring migration"],
            "year": [2020, 2020],
            "source_doi": ["test", "test"],
            "license": ["CC0-1.0", "CC0-1.0"],
        }
    )
    frame.to_parquet(telemetry_root / "part.parquet", index=False)
    database = rebuild_database(tmp_path)
    store = CatalogStore(database)
    populations = store.populations()
    assert populations.iloc[0].species == "Test bird"
    assert len(store.telemetry("Test bird", "Test Flyway", "spring migration")) == 2


def test_migration_queries_exclude_short_ranges_and_combine_seasons(tmp_path: Path):
    telemetry_root = tmp_path / "telemetry" / "study_id=1"
    rows = []
    for season, month, reverse in [("spring migration", 4, False), ("fall migration", 9, True)]:
        for animal in range(3):
            coordinates = [(25.0, -95.0), (30.0, -92.0), (36.0, -89.0), (42.0, -86.0)]
            if reverse:
                coordinates = list(reversed(coordinates))
            for point, (latitude, longitude) in enumerate(coordinates):
                rows.append(
                    {
                        "study_id": "1", "study_key": "test", "population": "Test Flyway",
                        "population_key": "test-flyway", "species": "Test bird",
                        "species_key": "test-bird", "animal_id": f"bird-{animal}",
                        "timestamp_utc": pd.Timestamp(2020, month, 1, tz="UTC") + pd.Timedelta(days=point * 4),
                        "latitude": latitude, "longitude": longitude, "sensor_type": "GPS",
                        "movement_type": "aerial", "hemisphere": "north", "season": season,
                        "year": 2020, "source_doi": "test", "license": "CC0-1.0",
                    }
                )
    frame = pd.DataFrame(rows)
    spring_only = frame[frame.season.eq("spring migration")].copy()
    spring_only["population"] = "Spring Only"
    spring_only["population_key"] = "spring-only"
    frame = pd.concat([frame, spring_only], ignore_index=True)
    destination = telemetry_root / "species_key=test-bird" / "population_key=test-flyway"
    destination.mkdir(parents=True)
    frame.to_parquet(destination / "part.parquet", index=False)
    store = CatalogStore(rebuild_database(tmp_path))
    populations = store.migration_populations()
    assert populations.iloc[0].journeys == 6
    assert "Spring Only" not in set(populations.population)
    telemetry = store.migration_telemetry("Test bird", "Test Flyway", max_journeys=6)
    assert set(telemetry.season) == {"spring migration", "fall migration"}


def test_manifest_taxa_have_child_friendly_names_and_groups():
    from core.taxonomy import load_taxonomy

    taxonomy = load_taxonomy()
    taxa = {taxon for study in load_manifest() for taxon in study.taxa}
    assert taxa <= set(taxonomy)
    assert taxonomy["Grus americana"] == {"common_name": "Whooping crane", "group": "Birds"}
