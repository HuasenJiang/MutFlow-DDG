from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import pickle
from typing import Any


GRAPH_KEYS = (
    "wild_type",
    "mutant",
    "wild_type_sequence",
    "mutant_sequence",
    "structure_scope",
    "sequence_scope",
)


def clean_name(value: Any) -> str:
    name = Path(str(value)).name
    return name[:-4] if name.lower().endswith(".pdb") else name


def identity(item: dict[str, Any]) -> tuple[str, str, float]:
    return (
        clean_name(item["wild_type_name"]),
        clean_name(item["mutant_name"]),
        float(item["ddg"]),
    )


def select_items(
    baseline_items: list[dict[str, Any]],
    pythia_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: dict[tuple[str, str, float], deque[dict[str, Any]]] = defaultdict(
        deque
    )
    for item in pythia_items:
        candidates[identity(item)].append(item)

    selected = []
    missing = []
    for index, baseline_item in enumerate(baseline_items):
        key = identity(baseline_item)
        if not candidates[key]:
            missing.append(
                {
                    "baseline_index": index,
                    "wild_type_name": key[0],
                    "mutant_name": key[1],
                    "ddg": key[2],
                }
            )
            continue
        pythia_item = candidates[key].popleft()
        merged = dict(baseline_item)
        for graph_key in GRAPH_KEYS:
            if graph_key in pythia_item:
                merged[graph_key] = pythia_item[graph_key]
        selected.append(merged)
    return selected, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select and order Pythia samples to match a CoMDDG pickle exactly."
        )
    )
    parser.add_argument("--baseline-pkl", required=True)
    parser.add_argument("--pythia-pkl", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

    with Path(args.baseline_pkl).open("rb") as handle:
        baseline_items = pickle.load(handle)
    with Path(args.pythia_pkl).open("rb") as handle:
        pythia_items = pickle.load(handle)

    selected, missing = select_items(baseline_items, pythia_items)
    report = {
        "baseline_pkl": args.baseline_pkl,
        "pythia_pkl": args.pythia_pkl,
        "output_pkl": args.output_pkl,
        "baseline_samples": len(baseline_items),
        "pythia_candidates": len(pythia_items),
        "selected_samples": len(selected),
        "missing_samples": len(missing),
        "missing": missing[:50],
    }
    if args.fail_on_missing and missing:
        raise RuntimeError(
            f"{len(missing)} baseline samples were absent from the Pythia pickle"
        )

    output_path = Path(args.output_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(selected, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "missing"}, indent=2))


if __name__ == "__main__":
    main()
