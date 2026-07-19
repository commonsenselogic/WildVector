from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.scenario_support import (  # noqa: E402
    validate_catalog_scenarios,
    write_scenario_support_registry,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify that every classroom experiment has an activated response."
    )
    parser.add_argument(
        "--catalog-root", type=Path, default=PROJECT_ROOT / "data" / "catalog"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = validate_catalog_scenarios(args.catalog_root)
    registry = write_scenario_support_registry(reports, args.catalog_root)
    for report in reports:
        status = "SUPPORTED" if report.supported else "MISSING"
        seasons = ", ".join(report.seasons) or "no activated season"
        outcomes = ", ".join(report.outcomes) or "no activated outcome"
        print(
            f"{status}: {report.species} / {report.scenario_key} / "
            f"{seasons} / {outcomes}"
        )
    unsupported = [report for report in reports if not report.supported]
    print(f"Scenario support registry: {registry}")
    if unsupported:
        print(
            "Classroom scenario validation failed. Retrain or revise the listed presets.",
            file=sys.stderr,
        )
        return 3
    print(f"All {len(reports)} classroom what-if scenarios are supported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
