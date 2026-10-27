from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_success_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "mutation_id" not in rows[0]:
        raise ValueError(f"{path} has no mutation_id column")
    return {
        row["mutation_id"]: row
        for row in rows
        if row.get("status", "success").lower()
        in {"success", "common", "included"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Intersect successful mutation IDs across methods and write the "
            "single common-subset manifest all models must use."
        )
    )
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument(
        "--union-datasets",
        action="store_true",
        help=(
            "Union manifests for disjoint dataset names. Use intersection "
            "(default) for multiple methods on the same dataset."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    paths = [Path(value) for value in args.manifests]
    by_method = [read_success_rows(path) for path in paths]
    common = set()
    if args.union_datasets:
        for rows in by_method:
            overlap = common.intersection(rows)
            if overlap:
                raise ValueError(
                    "union-datasets inputs overlap; intersect same-dataset "
                    "method manifests first"
                )
            common.update(rows)
    else:
        common = set(by_method[0]) if by_method else set()
        for rows in by_method[1:]:
            common.intersection_update(rows)
    source_by_id = {
        mutation_id: row for rows in by_method for mutation_id, row in rows.items()
    }
    ordered = sorted(
        common,
        key=lambda mutation_id: (
            source_by_id[mutation_id].get("dataset", ""),
            int(source_by_id[mutation_id].get("source_row_index") or 10**9),
        ),
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "mutation_id",
        "source_row_index",
        "protein_id",
        "chain",
        "mutation",
        "ddg",
        "status",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mutation_id in ordered:
            source = source_by_id[mutation_id]
            writer.writerow(
                {
                    **{name: source.get(name, "") for name in fieldnames},
                    "status": "common",
                }
            )

    report = {
        "output": str(output),
        "common_samples": len(ordered),
        "inputs": [
            {
                "manifest": str(path),
                "successful_samples": len(rows),
                "excluded_from_common": (
                    0 if args.union_datasets else len(rows) - len(ordered)
                ),
            }
            for path, rows in zip(paths, by_method)
        ],
    }
    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
