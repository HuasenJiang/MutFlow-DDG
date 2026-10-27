from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

from mutflow_ddg.data.identity import (
    baseline_rows,
    match_items_to_baseline,
    parse_record_name,
)


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "dataset",
        "mutation_id",
        "source_row_index",
        "input_index",
        "protein_id",
        "chain",
        "mutation",
        "ddg",
        "status",
        "reason",
        "recovery_source",
        "wild_type_name",
        "mutant_name",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [{name: row.get(name, "") for name in fieldnames} for row in rows]
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach stable baseline mutation IDs to an existing graph pickle."
    )
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--manifest-dir", required=True)
    args = parser.parse_args()

    path = Path(args.input_pkl)
    with path.open("rb") as handle:
        items = pickle.load(handle)
    reference = baseline_rows(args.baseline_csv, args.dataset_name)
    matches = match_items_to_baseline(items, reference)
    rows = []
    represented = set()
    for index, (item, matched) in enumerate(zip(items, matches)):
        parts = parse_record_name(item.get("mutant_name", ""))
        if matched is None:
            matched = {
                "dataset": args.dataset_name.lower(),
                "source_row_index": "",
                "protein_id": parts["protein_id"],
                "chain": parts["chain"],
                "mutation": parts["mutation"],
                "ddg": float(item["ddg"]),
                "mutation_id": (
                    f"{args.dataset_name.lower()}:input:{index:06d}:"
                    f"{parts['protein_id']}:{parts['chain']}:{parts['mutation']}"
                ),
            }
        item["mutation_id"] = matched["mutation_id"]
        item["source_row_index"] = matched["source_row_index"]
        represented.add(matched["mutation_id"])
        rows.append(
            {
                **matched,
                "input_index": index,
                "status": "success",
                "reason": "",
                "recovery_source": item.get(
                    "recovery_source", "converted_structure"
                ),
                "wild_type_name": item.get("wild_type_name", ""),
                "mutant_name": item.get("mutant_name", ""),
                "ddg": float(item["ddg"]),
            }
        )
    for reference_row in reference:
        if reference_row["mutation_id"] not in represented:
            rows.append(
                {
                    **reference_row,
                    "input_index": "",
                    "status": "failed",
                    "reason": "missing_from_input_pickle",
                    "recovery_source": "",
                    "wild_type_name": "",
                    "mutant_name": "",
                }
            )

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(items, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    manifest_dir = Path(args.manifest_dir)
    write_rows(manifest_dir / "all_samples.csv", rows)
    write_rows(
        manifest_dir / "success_samples.csv",
        [row for row in rows if row["status"] == "success"],
    )
    write_rows(
        manifest_dir / "failed_samples.csv",
        [row for row in rows if row["status"] == "failed"],
    )
    print(
        f"{args.dataset_name}: success={len(items)}, "
        f"failed_reference_rows={sum(row['status'] == 'failed' for row in rows)}"
    )


if __name__ == "__main__":
    main()
