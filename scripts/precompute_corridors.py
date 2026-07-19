from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.catalog import CatalogStore, rebuild_database  # noqa: E402
from core.population import build_population_corridor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute species-population-season corridors and representative journeys."
    )
    parser.add_argument(
        "--catalog-root", type=Path, default=PROJECT_ROOT / "data" / "catalog"
    )
    parser.add_argument("--bins", type=int, default=36)
    parser.add_argument("--max-journeys", type=int, default=250)
    return parser.parse_args()


def _atomic_parquet(frame: pd.DataFrame, path: Path):
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    store = CatalogStore(args.catalog_root / "wildvector.duckdb")
    corridor_rows = []
    journey_rows = []
    representative_rows = []
    populations = store.populations()
    for row in populations.itertuples(index=False):
        telemetry = store.telemetry(
            row.species, row.population, row.season, max_journeys=args.max_journeys
        )
        try:
            result = build_population_corridor(telemetry, bins=args.bins)
        except ValueError as exc:
            print(f"skip {row.species} / {row.population} / {row.season}: {exc}")
            continue
        identity = {
            "species": row.species,
            "population": row.population,
            "movement_type": row.movement_type,
            "season": row.season,
            "years": result.years,
            "animals": result.animals,
        }
        corridor_rows.append(result.corridor.assign(**identity))
        journey_rows.append(result.journeys.assign(**identity))
        representative_rows.append(
            result.paths[result.paths.representative].assign(**identity)
        )
        print(
            f"ready {row.species} / {row.population} / {row.season}: "
            f"{result.animals} animals, {result.years} years"
        )
    if not corridor_rows:
        print("No corridors could be built", file=sys.stderr)
        return 1
    args.catalog_root.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(pd.concat(corridor_rows, ignore_index=True), args.catalog_root / "corridors.parquet")
    _atomic_parquet(pd.concat(journey_rows, ignore_index=True), args.catalog_root / "journeys.parquet")
    _atomic_parquet(
        pd.concat(representative_rows, ignore_index=True),
        args.catalog_root / "representative-journeys.parquet",
    )
    rebuild_database(args.catalog_root)
    print(f"Precomputed {len(corridor_rows)} population-season corridors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
