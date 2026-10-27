from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from mutflow_ddg.metrics import regression_metrics


def _read_predictions(path: Path) -> dict[str, tuple[float, float]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"mutation_id", "target", "prediction"}
    missing = required - (set(rows[0]) if rows else set())
    if missing:
        raise ValueError(f"{path} missing prediction columns: {sorted(missing)}")
    result = {}
    for row in rows:
        if row.get("direction", "direct") != "direct":
            continue
        mutation_id = row["mutation_id"]
        if mutation_id in result:
            raise ValueError(f"duplicate direct prediction ID in {path}: {mutation_id}")
        result[mutation_id] = (float(row["prediction"]), float(row["target"]))
    return result


def _read_ids(path: Path) -> set[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["mutation_id"]
        for row in rows
        if row.get("status", "common").lower() in {"common", "success", "included"}
    }


def _metric_row(
    method: str,
    values: dict[str, tuple[float, float]],
    ids: set[str],
) -> dict[str, object]:
    missing = ids - set(values)
    if missing:
        examples = ", ".join(sorted(missing)[:3])
        raise ValueError(
            f"{method} lacks {len(missing)} required mutation IDs; examples: {examples}"
        )
    ordered = sorted(ids)
    prediction = torch.tensor([values[key][0] for key in ordered])
    target = torch.tensor([values[key][1] for key in ordered])
    return {"method": method, "samples": len(ordered), **regression_metrics(prediction, target)}


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["method", "samples", "mae", "mse", "rmse", "pcc", "spearman"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write full-set baseline and common-subset comparison to separate "
            "files so the two evaluation protocols cannot be mixed."
        )
    )
    parser.add_argument("--baseline", required=True, help="NAME=predictions.csv")
    parser.add_argument("--methods", nargs="+", required=True, help="NAME=predictions.csv")
    parser.add_argument("--common-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    def parse_spec(value: str) -> tuple[str, Path]:
        if "=" not in value:
            raise ValueError(f"prediction specification must be NAME=PATH: {value}")
        name, path = value.split("=", 1)
        return name, Path(path)

    baseline_name, baseline_path = parse_spec(args.baseline)
    method_specs = [parse_spec(value) for value in args.methods]
    predictions = {
        baseline_name: _read_predictions(baseline_path),
        **{name: _read_predictions(path) for name, path in method_specs},
    }
    common_ids = _read_ids(Path(args.common_manifest))
    if not common_ids:
        raise ValueError("common-subset manifest is empty")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_ids = set(predictions[baseline_name])
    full_row = _metric_row(baseline_name, predictions[baseline_name], full_ids)
    common_rows = [
        _metric_row(name, values, common_ids)
        for name, values in predictions.items()
    ]
    _write(output_dir / "full_set_baseline_metrics.csv", [full_row])
    _write(output_dir / "common_subset_metrics.csv", common_rows)
    report = {
        "full_set_baseline": {
            "file": str(output_dir / "full_set_baseline_metrics.csv"),
            **full_row,
        },
        "common_subset_comparison": {
            "file": str(output_dir / "common_subset_metrics.csv"),
            "samples": len(common_ids),
            "methods": [row["method"] for row in common_rows],
        },
    }
    (output_dir / "protocol_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
