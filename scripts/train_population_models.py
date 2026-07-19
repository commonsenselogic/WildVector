from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.catalog import CatalogStore  # noqa: E402
from core.classroom_catalog import filter_classroom_populations  # noqa: E402
from core.outcome_model import (  # noqa: E402
    extract_journey_outcomes,
    fit_outcome_models,
    save_outcome_models,
)
from core.scenario_support import (  # noqa: E402
    validate_catalog_scenarios,
    write_scenario_support_registry,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Model migration outcomes with forward-held-out years and seasonal/persistence baselines."
        )
    )
    parser.add_argument(
        "--catalog-root", type=Path, default=PROJECT_ROOT / "data" / "catalog"
    )
    parser.add_argument("--species", help="Optional exact scientific-name filter.")
    parser.add_argument("--season", choices=["spring migration", "fall migration"])
    parser.add_argument("--max-journeys", type=int, default=500)
    parser.add_argument(
        "--all-populations", action="store_true",
        help="Train every qualifying catalog population instead of the three classroom animals.",
    )
    return parser.parse_args()


def _write_registry(catalog_root: Path) -> Path:
    model_root = catalog_root / "outcome-models"
    validations = []
    for path in sorted(model_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) < 2:
            continue
        validations.extend(payload.get("validations", []))
    source_trials = []
    for path in sorted(model_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_trials.extend(payload.get("source_trials", []))
    registry = {
        "schema_version": 4,
        "activation_rule": {
            "minimum_forward_folds": 3,
            "minimum_skill_vs_each_baseline": 0.05,
            "minimum_fold_win_rate": 0.60,
            "minimum_effect_sign_stability": 0.67,
        },
        "validations": validations,
        "source_trials": source_trials,
        "included_sources": [
            row for row in source_trials if row.get("model_status") == "included"
        ],
        "active": [
            row for row in validations if row.get("status") == "active historical association"
        ],
    }
    path = catalog_root / "outcome-activation.json"
    temporary = catalog_root / "outcome-activation.tmp.json"
    temporary.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def main() -> int:
    args = parse_args()
    journey_environment_path = args.catalog_root / "environment-journeys.parquet"
    environment_path = (
        journey_environment_path
        if journey_environment_path.exists()
        else args.catalog_root / "environment.parquet"
    )
    if not environment_path.exists():
        print("Build environmental history first with scripts/refresh_environment.py", file=sys.stderr)
        return 2
    environment = pd.read_parquet(environment_path)
    store = CatalogStore(args.catalog_root / "wildvector.duckdb")
    populations = store.migration_populations()
    if not args.all_populations:
        populations = filter_classroom_populations(populations)
    if args.species:
        populations = populations[populations.species.eq(args.species)]
    trained = 0
    for population in populations.itertuples(index=False):
        available_seasons = environment[
            environment.species.eq(population.species)
            & environment.population.eq(population.population)
        ].season.unique()
        if len(available_seasons) == 0:
            continue
        combined = store.migration_telemetry(
            population.species, population.population, max_journeys=args.max_journeys
        )
        for season in ("spring migration", "fall migration"):
            if args.season and season != args.season:
                continue
            if season not in available_seasons:
                continue
            telemetry = combined[combined.season.eq(season)].copy()
            subset = environment[
                environment.species.eq(population.species)
                & environment.population.eq(population.population)
                & environment.season.eq(season)
            ].copy()
            if "journey_id" in subset:
                expected = set(extract_journey_outcomes(telemetry).journey_id.astype(str))
                complete_counts = subset.groupby("journey_id").window_name.nunique()
                complete = set(complete_counts[complete_counts >= 6].index.astype(str))
                matched = expected & complete
                coverage = len(matched) / len(expected) if expected else 0.0
                matched_years = int(
                    subset[subset.journey_id.astype(str).isin(matched)].year.nunique()
                )
                if len(matched) < 12 or matched_years < 4:
                    print(
                        f"skip {population.species} / {population.population} / {season}: "
                        f"only {len(matched)} complete journeys across {matched_years} years "
                        f"({coverage:.1%} of telemetry)"
                    )
                    continue
                if coverage < 0.95:
                    print(
                        f"partial {population.species} / {population.population} / {season}: "
                        f"training on {len(matched)} complete journeys across {matched_years} years "
                        f"({coverage:.1%} of telemetry)"
                    )
            if not subset.empty:
                telemetry = telemetry[telemetry.year.isin(subset.year.unique())]
            try:
                bundle, validations = fit_outcome_models(telemetry, subset)
            except ValueError as exc:
                print(f"skip {population.species} / {population.population} / {season}: {exc}")
                continue
            save_outcome_models(bundle, validations, args.catalog_root)
            trained += 1
            active = [row.outcome for row in validations if row.status == "active historical association"]
            print(
                f"{population.species} / {population.population} / {season}: "
                f"{len(validations)} outcomes, active={active or 'none'}"
            )
    registry = _write_registry(args.catalog_root)
    print(f"Trained {trained} population-season outcome bundles")
    print(f"Activation registry: {registry}")
    if not args.species and not args.season:
        scenario_reports = validate_catalog_scenarios(args.catalog_root)
        scenario_registry = write_scenario_support_registry(
            scenario_reports, args.catalog_root
        )
        unsupported = [report for report in scenario_reports if not report.supported]
        print(f"Classroom scenario registry: {scenario_registry}")
        if unsupported:
            for report in unsupported:
                print(
                    f"unsupported classroom scenario: {report.species} / "
                    f"{report.scenario_key}",
                    file=sys.stderr,
                )
            return 3
        print(f"All {len(scenario_reports)} classroom what-if scenarios are supported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
