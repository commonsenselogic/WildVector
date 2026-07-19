from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.catalog import CatalogStore, rebuild_database  # noqa: E402
from core.classroom_catalog import filter_classroom_populations  # noqa: E402
from core.environmental_history import (  # noqa: E402
    EnvironmentError,
    build_journey_weather_points,
    fetch_journey_atmosphere_history,
    write_journey_environment,
)
from core.predictor_enrichment import (  # noqa: E402
    SourceResult,
    fetch_erddap_marine_features,
    fetch_fox_sea_ice_features,
    fetch_glorys_journey_features,
    fetch_hourly_land_features,
    join_bylot_lemmings,
    load_bylot_lemming_index,
    merge_feature_frames,
)


ENRICHMENT_SOURCES = (
    "hourly-land", "bylot-lemmings", "sea-ice", "glorys", "marine-biology"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build resumable, journey-specific multi-year ERA5 weather history."
    )
    parser.add_argument(
        "--catalog-root", type=Path, default=PROJECT_ROOT / "data" / "catalog"
    )
    parser.add_argument("--max-journeys", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--species", help="Optional exact scientific-name filter for a staged build.")
    parser.add_argument("--season", choices=["spring migration", "fall migration"])
    parser.add_argument(
        "--max-years", type=int,
        help="Optional number of most recent observed season-years; default keeps every year.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Refresh completed journeys too. By default the job resumes around completed data.",
    )
    parser.add_argument(
        "--all-populations", action="store_true",
        help="Refresh every qualifying catalog population instead of the three classroom animals.",
    )
    parser.add_argument(
        "--marine", action="store_true",
        help="Backward-compatible alias enabling GLORYS and marine biology enrichment.",
    )
    parser.add_argument(
        "--source", action="append", choices=ENRICHMENT_SOURCES,
        help=(
            "Enrichment to run after ERA5; repeat for multiple sources. Default runs open, "
            "species-relevant sources and attempts GLORYS only with --marine or --source glorys."
        ),
    )
    return parser.parse_args()


def _write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)


def _completed_journeys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    existing = pd.read_parquet(path, columns=["journey_id", "window_name"])
    counts = existing.groupby("journey_id").window_name.nunique()
    return set(counts[counts >= 6].index.astype(str))


def main() -> int:
    args = parse_args()
    store = CatalogStore(args.catalog_root / "wildvector.duckdb")
    populations = store.migration_populations()
    if not args.all_populations:
        populations = filter_classroom_populations(populations)
    if args.species:
        populations = populations[populations.species.eq(args.species)]
    output_path = args.catalog_root / "environment-journeys.parquet"
    report_path = args.catalog_root / "environment-refresh-report.json"
    report = {
        "schema_version": 2,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": "Open-Meteo ERA5 daily reanalysis",
        "max_journeys": args.max_journeys,
        "groups": [],
    }
    completed = set() if args.force else _completed_journeys(output_path)
    refreshed = failures = skipped = 0
    selected_sources = set(args.source or (
        "hourly-land", "bylot-lemmings", "sea-ice", "marine-biology"
    ))
    if args.marine:
        selected_sources.update({"glorys", "marine-biology"})
    report["enrichment_sources"] = sorted(selected_sources)
    source_results: list[SourceResult] = []
    lemmings = None
    if "bylot-lemmings" in selected_sources:
        try:
            lemmings = load_bylot_lemming_index(args.catalog_root)
            source_results.append(SourceResult("bylot-lemmings", "ready", len(lemmings)))
        except EnvironmentError as exc:
            source_results.append(SourceResult("bylot-lemmings", "failed", 0, str(exc)))

    for population in populations.itertuples(index=False):
        telemetry = store.migration_telemetry(
            population.species, population.population, max_journeys=args.max_journeys
        )
        for season in ("spring migration", "fall migration"):
            if args.season and season != args.season:
                continue
            selected = telemetry[telemetry.season.eq(season)].copy()
            if args.max_years and not selected.empty:
                years = sorted(selected.year.unique())[-args.max_years :]
                selected = selected[selected.year.isin(years)]
            if selected.empty:
                continue
            points = build_journey_weather_points(selected)
            if not args.force and completed:
                points = points[~points.journey_id.isin(completed)]
            identity = f"{population.species} / {population.population} / {season}"
            if points.empty:
                print(f"resume {identity}: already complete")
                skipped += 1
                continue
            print(
                f"queue {identity}: {points.journey_id.nunique()} journeys, "
                f"{points.year.nunique()} years, {len(points)} windows",
                flush=True,
            )
            # A completed year is immediately durable. If a public API quota interrupts
            # a large species, the next run resumes without downloading prior years again.
            for year, year_points in points.groupby("year", sort=True):
                year_identity = f"{identity} / {int(year)}"
                journeys = int(year_points.journey_id.nunique())
                entry = {
                    "species": population.species,
                    "population": population.population,
                    "season": season,
                    "year": int(year),
                    "journeys": journeys,
                    "windows_requested": len(year_points),
                }
                print(f"fetch {year_identity}: {journeys} journeys, {len(year_points)} windows", flush=True)
                try:
                    weather = fetch_journey_atmosphere_history(
                        year_points, batch_size=args.batch_size
                    )
                    feature_frames = []
                    if "hourly-land" in selected_sources and population.movement_type != "marine":
                        try:
                            hourly = fetch_hourly_land_features(year_points)
                            feature_frames.append(hourly)
                            source_results.append(SourceResult("hourly-land", "ready", len(hourly)))
                        except EnvironmentError as exc:
                            source_results.append(SourceResult("hourly-land", "failed", 0, str(exc)))
                    if lemmings is not None and population.species == "Vulpes lagopus":
                        feature_frames.append(join_bylot_lemmings(year_points, lemmings))
                    if "sea-ice" in selected_sources and population.species == "Vulpes lagopus":
                        try:
                            sea_ice = fetch_fox_sea_ice_features(year_points)
                            feature_frames.append(sea_ice)
                            source_results.append(SourceResult("sea-ice", "ready", len(sea_ice)))
                        except EnvironmentError as exc:
                            source_results.append(SourceResult("sea-ice", "failed", 0, str(exc)))
                    if "glorys" in selected_sources and population.movement_type == "marine":
                        try:
                            glorys = fetch_glorys_journey_features(year_points)
                            feature_frames.append(glorys)
                            source_results.append(SourceResult("glorys", "ready", len(glorys)))
                        except EnvironmentError as exc:
                            source_results.append(SourceResult("glorys", "unavailable", 0, str(exc)))
                    if "marine-biology" in selected_sources and population.movement_type == "marine":
                        try:
                            biology = fetch_erddap_marine_features(year_points)
                            feature_frames.append(biology)
                            source_results.append(SourceResult("marine-biology", "ready", len(biology)))
                        except EnvironmentError as exc:
                            source_results.append(SourceResult("marine-biology", "failed", 0, str(exc)))
                    weather = merge_feature_frames(weather, feature_frames)
                    write_journey_environment(weather, args.catalog_root, merge_existing=True)
                    completed.update(weather.journey_id.astype(str).unique())
                    entry.update(status="complete", rows=len(weather))
                    refreshed += 1
                    print(f"saved {year_identity}: {len(weather)} weather windows", flush=True)
                except (EnvironmentError, OSError, ValueError) as exc:
                    entry.update(status="failed", error=str(exc))
                    failures += 1
                    print(f"failed {year_identity}: {exc}", file=sys.stderr, flush=True)
                report["groups"].append(entry)
                report["updated_at"] = datetime.now(timezone.utc).isoformat()
                _write_report(report_path, report)

    report.update(
        finished_at=datetime.now(timezone.utc).isoformat(),
        refreshed_groups=refreshed,
        failed_groups=failures,
        resumed_groups=skipped,
        source_results=[result.__dict__ for result in source_results],
    )
    _write_report(report_path, report)
    if output_path.exists():
        rebuild_database(args.catalog_root)
        stored = pd.read_parquet(output_path, columns=["journey_id", "year", "window_name"])
        print(
            f"Journey environment: {output_path} ({stored.journey_id.nunique():,} journeys, "
            f"{stored.year.nunique()} years, {len(stored):,} windows)"
        )
    print(f"Refresh report: {report_path}")
    return 1 if failures and not refreshed else 0


if __name__ == "__main__":
    raise SystemExit(main())
