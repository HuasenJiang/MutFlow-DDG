from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

from mutflow_ddg.data.identity import baseline_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Attach baseline IDs using a verified sample_index -> source_row_index "
            "bridge manifest (useful when two releases rename proteins/chains)."
        )
    )
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--index-manifest", required=True)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--output-manifest-dir", required=True)
    args = parser.parse_args()

    with Path(args.input_pkl).open("rb") as handle:
        items = pickle.load(handle)
    with Path(args.index_manifest).open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        index_rows = list(csv.DictReader(handle))
    reference = baseline_rows(args.baseline_csv, args.dataset_name)
    if len(index_rows) != len(items):
        raise ValueError(
            f"index manifest/items mismatch: {len(index_rows)} vs {len(items)}"
        )

    success = []
    represented = set()
    for index, (item, bridge) in enumerate(zip(items, index_rows)):
        if int(bridge["sample_index"]) != index:
            raise ValueError("index manifest is not aligned to pickle ordering")
        source_index = int(bridge["source_row_index"])
        row = reference[source_index]
        item["mutation_id"] = row["mutation_id"]
        item["source_row_index"] = source_index
        represented.add(row["mutation_id"])
        success.append(
            {
                **row,
                "input_index": index,
                "status": "success",
                "reason": "",
                "wild_type_name": item.get("wild_type_name", ""),
                "mutant_name": item.get("mutant_name", ""),
            }
        )
    failed = [
        {
            **row,
            "input_index": "",
            "status": "failed",
            "reason": "missing_from_pythia_graphs",
            "wild_type_name": "",
            "mutant_name": "",
        }
        for row in reference
        if row["mutation_id"] not in represented
    ]

    output = Path(args.output_pkl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(items, handle, protocol=pickle.HIGHEST_PROTOCOL)
    fields = [
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
        "wild_type_name",
        "mutant_name",
    ]
    manifest_dir = Path(args.output_manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("all_samples.csv", [*success, *failed]),
        ("success_samples.csv", success),
        ("failed_samples.csv", failed),
    ):
        with (manifest_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                [{field: row.get(field, "") for field in fields} for row in rows]
            )
    print(
        f"{args.dataset_name}: success={len(success)}, failed={len(failed)}"
    )


if __name__ == "__main__":
    main()
