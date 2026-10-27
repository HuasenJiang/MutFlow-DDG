from __future__ import annotations

import argparse
import json
import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PDB_ID_RE = re.compile(r"^([0-9][A-Za-z0-9]{3})")


@dataclass(frozen=True)
class RecordLocation:
    shard_id: int
    local_index: int
    order: int


def _resolve_shards(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.pkl"))
    return [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record_name(record: dict[str, Any], fallback: str) -> str:
    return Path(str(record.get("name", fallback))).name


def _group_key(
    record: dict[str, Any],
    fallback: str,
    group_by: str,
) -> str:
    name = _record_name(record, fallback)
    if group_by == "none":
        return fallback
    if group_by == "name":
        return name

    match = PDB_ID_RE.match(name)
    if match:
        return match.group(1).upper()
    stem = Path(name).stem
    return re.split(r"[_\-.]", stem, maxsplit=1)[0].upper()


def _collect_groups(
    shard_paths: list[Path],
    group_by: str,
) -> dict[str, list[RecordLocation]]:
    groups: dict[str, list[RecordLocation]] = defaultdict(list)
    order = 0
    for shard_id, shard_path in enumerate(shard_paths):
        with shard_path.open("rb") as handle:
            records = pickle.load(handle)
        for local_index, record in enumerate(records):
            fallback = f"{shard_id}:{local_index}"
            key = _group_key(record, fallback, group_by)
            groups[key].append(
                RecordLocation(
                    shard_id=shard_id,
                    local_index=local_index,
                    order=order,
                )
            )
            order += 1
    return groups


def _assign_groups(
    groups: dict[str, list[RecordLocation]],
    validation_size: int,
    test_size: int,
    seed: int,
) -> dict[str, str]:
    if validation_size < 0 or test_size < 0:
        raise ValueError("validation_size and test_size must be non-negative")
    total_records = sum(len(locations) for locations in groups.values())
    if validation_size + test_size >= total_records:
        raise ValueError(
            "validation_size + test_size must be smaller than total records"
        )

    group_keys = sorted(groups)
    random.Random(seed).shuffle(group_keys)
    assignments: dict[str, str] = {}
    counts = {"validation": 0, "test": 0}

    for key in group_keys:
        group_size = len(groups[key])
        if counts["validation"] < validation_size:
            assignments[key] = "validation"
            counts["validation"] += group_size
        elif counts["test"] < test_size:
            assignments[key] = "test"
            counts["test"] += group_size
        else:
            assignments[key] = "train"
    return assignments


def _flush_split_shard(
    output_root: Path,
    prefix: str,
    split: str,
    shard_index: int,
    records: list[dict[str, Any]],
) -> Path:
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"{prefix}_{split}_graphs_{shard_index:04d}.pkl"
    with path.open("wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _write_split_records(
    shard_paths: list[Path],
    groups: dict[str, list[RecordLocation]],
    assignments: dict[str, str],
    output_root: Path,
    prefix: str,
    shard_size: int,
    group_by: str,
) -> dict[str, list[str]]:
    if shard_size < 1:
        raise ValueError("shard_size must be positive")

    location_to_split: dict[tuple[int, int], str] = {}
    for key, locations in groups.items():
        split = assignments[key]
        for location in locations:
            location_to_split[(location.shard_id, location.local_index)] = split

    buffers: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    shards: dict[str, list[str]] = {"train": [], "validation": [], "test": []}

    for shard_id, shard_path in enumerate(shard_paths):
        with shard_path.open("rb") as handle:
            records = pickle.load(handle)
        for local_index, record in enumerate(records):
            split = location_to_split[(shard_id, local_index)]
            buffers[split].append(record)
            if len(buffers[split]) >= shard_size:
                path = _flush_split_shard(
                    output_root,
                    prefix,
                    split,
                    len(shards[split]),
                    buffers[split],
                )
                shards[split].append(str(path))
                buffers[split] = []

    for split, records in buffers.items():
        if records:
            path = _flush_split_shard(
                output_root,
                prefix,
                split,
                len(shards[split]),
                records,
            )
            shards[split].append(str(path))

    for split, split_shards in shards.items():
        manifest = output_root / split / f"{prefix}_{split}_graphs_shards.txt"
        manifest.write_text("\n".join(split_shards) + "\n", encoding="utf-8")
    return shards


def _split_counts(
    groups: dict[str, list[RecordLocation]],
    assignments: dict[str, str],
) -> dict[str, dict[str, int]]:
    counts = {
        "train": {"groups": 0, "records": 0},
        "validation": {"groups": 0, "records": 0},
        "test": {"groups": 0, "records": 0},
    }
    for key, split in assignments.items():
        counts[split]["groups"] += 1
        counts[split]["records"] += len(groups[key])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split successfully converted structure graph shards."
    )
    parser.add_argument("--input", required=True, help="Shard manifest or shard dir.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="rcsb_bioa")
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument(
        "--group-by",
        choices=("pdb", "name", "none"),
        default="pdb",
        help="Group records before splitting to avoid leakage. Default: pdb.",
    )
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output_dir)
    shard_paths = _resolve_shards(input_path)
    if not shard_paths:
        raise ValueError(f"No input shards found under {input_path}")

    groups = _collect_groups(shard_paths, args.group_by)
    assignments = _assign_groups(
        groups,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    split_shards = _write_split_records(
        shard_paths,
        groups,
        assignments,
        output_root,
        prefix=args.prefix,
        shard_size=args.shard_size,
        group_by=args.group_by,
    )
    counts = _split_counts(groups, assignments)

    report = {
        "input": str(input_path),
        "output_dir": str(output_root),
        "seed": args.seed,
        "group_by": args.group_by,
        "validation_size_requested": args.validation_size,
        "test_size_requested": args.test_size,
        "total_groups": len(groups),
        "total_records": sum(split["records"] for split in counts.values()),
        "splits": counts,
        "shards": split_shards,
        "manifests": {
            split: str(output_root / split / f"{args.prefix}_{split}_graphs_shards.txt")
            for split in split_shards
        },
    }
    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
