from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

from mutflow_ddg.data.comddg import ESMFeatureStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write success/failure manifests for paired ESM availability."
    )
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--cache-dir", action="append", required=True)
    parser.add_argument("--layout", choices=("standard", "paired_flat"), default="standard")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with Path(args.input_pkl).open("rb") as handle:
        items = pickle.load(handle)
    store = ESMFeatureStore(args.cache_dir, layout=args.layout)
    rows = []
    for index, item in enumerate(items):
        paths = store.paths_for_item(item)
        missing_sides = [
            side
            for side, path in zip(("wild_type", "mutant"), paths)
            if path is None
        ]
        length_mismatches = []
        for side, path in zip(("wild_type", "mutant"), paths):
            if path is None:
                continue
            record = item.get(f"{side}_sequence", item[side])
            tensor = store._load_path(path)
            expected_length = len(record["node_features"])
            if tensor.shape != (expected_length, 1280):
                length_mismatches.append(
                    f"{side}:{tuple(tensor.shape)}!={(expected_length, 1280)}"
                )
            store._cache.clear()
        failed = bool(missing_sides or length_mismatches)
        rows.append(
            {
                "dataset": args.dataset_name.lower(),
                "mutation_id": item.get("mutation_id", ""),
                "source_row_index": item.get("source_row_index", ""),
                "input_index": index,
                "status": "failed" if failed else "success",
                "reason": (
                    "missing_esm:" + ",".join(missing_sides)
                    if missing_sides
                    else (
                        "esm_length_mismatch:" + ";".join(length_mismatches)
                        if length_mismatches
                        else ""
                    )
                ),
                "wild_type_name": item.get("wild_type_name", ""),
                "mutant_name": item.get("mutant_name", ""),
                "ddg": item.get("ddg", ""),
            }
        )
    fieldnames = list(rows[0]) if rows else ["dataset", "mutation_id", "status"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, selected in (
        ("all_samples.csv", rows),
        ("success_samples.csv", [row for row in rows if row["status"] == "success"]),
        ("failed_samples.csv", [row for row in rows if row["status"] == "failed"]),
    ):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)
    print(
        f"{args.dataset_name}: ESM success="
        f"{sum(row['status'] == 'success' for row in rows)}, "
        f"failed={sum(row['status'] == 'failed' for row in rows)}"
    )


if __name__ == "__main__":
    main()
