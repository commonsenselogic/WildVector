from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.catalog import (  # noqa: E402
    CatalogError,
    MovebankRepositoryClient,
    RefreshResult,
    load_manifest,
    rebuild_database,
    refresh_study,
    write_refresh_report,
)
from core.classroom_catalog import classroom_study_keys  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build WildVector's licensed local Movebank Parquet catalog."
    )
    parser.add_argument(
        "--study",
        action="append",
        default=[],
        help="Study key to refresh. Repeat to select several; omit for the three classroom studies.",
    )
    parser.add_argument(
        "--all-studies",
        action="store_true",
        help="Refresh all 30 expansion studies; the default refreshes the three classroom studies.",
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "catalog",
    )
    parser.add_argument(
        "--max-file-mb",
        type=int,
        default=750,
        help="Safety limit for each repository CSV download.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed study and continue refreshing the remaining selection.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate the manifest and rebuild DuckDB metadata without downloads.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    studies = load_manifest()
    by_key = {study.key: study for study in studies}
    unknown = sorted(set(args.study) - set(by_key))
    if unknown:
        print(f"Unknown study key(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    if args.study:
        selected = [by_key[key] for key in args.study]
    elif args.all_studies:
        selected = studies
    else:
        selected = [by_key[key] for key in classroom_study_keys()]
    results: list[RefreshResult] = []
    if args.metadata_only:
        client = MovebankRepositoryClient()
        for index, study in enumerate(selected, start=1):
            print(f"[{index}/{len(selected)}] {study.key}: validating metadata")
            try:
                client.validate(study)
                results.append(
                    RefreshResult(study.key, "metadata_valid", 0, 0, 0, 0, 0, 0)
                )
            except CatalogError as exc:
                results.append(
                    RefreshResult(study.key, "failed", 0, 0, 0, 0, 0, 0, str(exc))
                )
                print(f"  failed: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    write_refresh_report(results, args.catalog_root)
                    return 1
    else:
        for index, study in enumerate(selected, start=1):
            print(f"[{index}/{len(selected)}] {study.key}: validating and downloading")
            try:
                result = refresh_study(
                    study,
                    catalog_root=args.catalog_root,
                    maximum_file_bytes=args.max_file_mb * 1_000_000,
                )
                print(
                    f"  ready: {result.rows:,} rows, {result.animals:,} animals, "
                    f"{result.years} years"
                )
            except CatalogError as exc:
                result = RefreshResult(study.key, "failed", 0, 0, 0, 0, 0, 0, str(exc))
                print(f"  failed: {exc}", file=sys.stderr)
                results.append(result)
                if not args.continue_on_error:
                    write_refresh_report(results, args.catalog_root)
                    return 1
                continue
            results.append(result)
    database = rebuild_database(args.catalog_root)
    write_refresh_report(results, args.catalog_root)
    print(f"Catalog database: {database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
