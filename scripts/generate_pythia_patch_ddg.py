from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASETS = (
    "s2648",
    "s250",
    "s669",
    "s605",
    "s1925",
    "p53",
    "s350",
    "ssym",
)


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value)


def _empty_edge_index(edge_index: np.ndarray) -> np.ndarray:
    return np.empty((2, 0), dtype=edge_index.dtype)


def _empty_edge_features(edge_features: np.ndarray) -> np.ndarray:
    width = edge_features.shape[1] if edge_features.ndim == 2 else 0
    return np.empty((0, width), dtype=edge_features.dtype)


def _select_ca_neighbors(
    record: dict[str, Any],
    center: int,
    neighbors: int,
    *,
    allow_sequence_fallback: bool = False,
) -> np.ndarray:
    node_features = _as_array(record["node_features"])
    length = node_features.shape[0]
    if center < 0 or center >= length:
        raise ValueError(
            f"mutation_pos {center} is outside graph length {length}"
        )

    ca_coords = record.get("ca_coords")
    if ca_coords is None:
        if not allow_sequence_fallback:
            raise ValueError(
                "Pythia patch generation requires ca_coords; pass "
                "--allow-sequence-fallback to use sequence-neighbor patches"
            )
        residue_distance = np.abs(np.arange(length, dtype=np.int64) - center)
        order = np.argsort(residue_distance, kind="stable")
    else:
        ca = _as_array(ca_coords).astype(np.float32, copy=False)
        if ca.shape != (length, 3):
            raise ValueError(
                "ca_coords and node_features disagree: "
                f"{ca.shape} vs ({length}, 3)"
            )
        distances = np.linalg.norm(ca - ca[center], axis=-1)
        order = np.argsort(distances, kind="stable")

    selected = order[: min(neighbors + 1, length)]
    if center not in set(selected.tolist()):
        selected = np.concatenate([selected, np.asarray([center], dtype=selected.dtype)])
    return np.sort(np.unique(selected)).astype(np.int64)


def _slice_node_value(value: Any, selected: np.ndarray, length: int) -> Any:
    if isinstance(value, np.ndarray) and value.shape[:1] == (length,):
        return value[selected]
    if isinstance(value, str) and len(value) == length:
        return "".join(value[int(index)] for index in selected)
    if isinstance(value, list) and len(value) == length:
        return [value[int(index)] for index in selected]
    if isinstance(value, tuple) and len(value) == length:
        return tuple(value[int(index)] for index in selected)
    return value


def crop_structure_record(
    record: dict[str, Any],
    selected: np.ndarray,
    *,
    center: int,
    neighbors: int,
) -> dict[str, Any]:
    node_features = _as_array(record["node_features"])
    edge_index = _as_array(record["edge_index"]).astype(np.int64, copy=False)
    edge_features = _as_array(record["edge_features"])
    length = node_features.shape[0]
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_features.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_features and edge_index disagree on num_edges")

    remap = np.full(length, -1, dtype=np.int64)
    remap[selected] = np.arange(selected.shape[0], dtype=np.int64)
    edge_mask = (remap[edge_index[0]] >= 0) & (remap[edge_index[1]] >= 0)

    patch = {
        key: _slice_node_value(value, selected, length)
        for key, value in record.items()
        if key not in {"edge_index", "edge_features", "mutation_pos"}
    }
    patch["edge_index"] = remap[edge_index[:, edge_mask]].astype(
        edge_index.dtype,
        copy=False,
    )
    patch["edge_features"] = edge_features[edge_mask]
    patch["mutation_pos"] = int(remap[center])
    patch["full_mutation_pos"] = int(center)
    patch["patch_source_indices"] = selected
    patch["patch_neighbors"] = int(neighbors)
    return patch


def make_sequence_record(record: dict[str, Any]) -> dict[str, Any]:
    node_features = _as_array(record["node_features"])
    edge_index = _as_array(record["edge_index"])
    edge_features = _as_array(record["edge_features"])
    keep_keys = {
        "node_features",
        "mutation_pos",
        "residue_type",
        "sequence",
        "name",
        "source_pdb",
        "chain_ids",
        "chain_index",
        "resseq",
    }
    sequence_record = {
        key: value
        for key, value in record.items()
        if key in keep_keys
    }
    sequence_record["node_features"] = node_features
    sequence_record["edge_index"] = _empty_edge_index(edge_index)
    sequence_record["edge_features"] = _empty_edge_features(edge_features)
    sequence_record["mutation_pos"] = int(record.get("mutation_pos", 0))
    sequence_record["sequence_scope"] = "full"
    return sequence_record


def convert_item(
    item: dict[str, Any],
    *,
    neighbors: int,
    allow_sequence_fallback: bool,
) -> dict[str, Any]:
    wild_full = item["wild_type"]
    mutant_full = item["mutant"]
    wild_length = len(wild_full["node_features"])
    mutant_length = len(mutant_full["node_features"])
    if wild_length != mutant_length:
        raise ValueError(
            "WT and mutant graph lengths differ: "
            f"{wild_length} vs {mutant_length}"
        )

    wild_center = int(wild_full.get("mutation_pos", 0))
    mutant_center = int(mutant_full.get("mutation_pos", wild_center))
    if wild_center != mutant_center:
        raise ValueError(
            "WT and mutant mutation_pos differ before patching: "
            f"{wild_center} vs {mutant_center}"
        )
    selected = _select_ca_neighbors(
        wild_full,
        wild_center,
        neighbors,
        allow_sequence_fallback=allow_sequence_fallback,
    )
    converted = dict(item)
    converted["wild_type_sequence"] = make_sequence_record(wild_full)
    converted["mutant_sequence"] = make_sequence_record(mutant_full)
    converted["wild_type"] = crop_structure_record(
        wild_full,
        selected,
        center=wild_center,
        neighbors=neighbors,
    )
    converted["mutant"] = crop_structure_record(
        mutant_full,
        selected,
        center=mutant_center,
        neighbors=neighbors,
    )
    converted["structure_scope"] = f"ca_patch{neighbors}"
    converted["sequence_scope"] = "full"
    return converted


def validate_converted_item(
    item: dict[str, Any],
    *,
    neighbors: int,
) -> dict[str, int]:
    wild_patch = item["wild_type"]
    mutant_patch = item["mutant"]
    wild_sequence = item["wild_type_sequence"]
    mutant_sequence = item["mutant_sequence"]
    patch_limit = neighbors + 1

    wild_patch_len = len(wild_patch["node_features"])
    mutant_patch_len = len(mutant_patch["node_features"])
    wild_sequence_len = len(wild_sequence["node_features"])
    mutant_sequence_len = len(mutant_sequence["node_features"])
    wild_patch_index = int(wild_patch["mutation_pos"])
    mutant_patch_index = int(mutant_patch["mutation_pos"])
    wild_sequence_index = int(wild_sequence["mutation_pos"])
    mutant_sequence_index = int(mutant_sequence["mutation_pos"])

    if wild_patch_len > patch_limit or mutant_patch_len > patch_limit:
        raise ValueError(
            f"patch contains more than {patch_limit} nodes: "
            f"{wild_patch_len}, {mutant_patch_len}"
        )
    for length, index, label in (
        (wild_patch_len, wild_patch_index, "wild_type"),
        (mutant_patch_len, mutant_patch_index, "mutant"),
        (wild_sequence_len, wild_sequence_index, "wild_type_sequence"),
        (mutant_sequence_len, mutant_sequence_index, "mutant_sequence"),
    ):
        if index < 0 or index >= length:
            raise ValueError(
                f"{label}.mutation_pos {index} outside length {length}"
            )
    if wild_sequence_index != int(wild_patch["full_mutation_pos"]):
        raise ValueError("WT full and patch mutation indices disagree")
    if mutant_sequence_index != int(mutant_patch["full_mutation_pos"]):
        raise ValueError("mutant full and patch mutation indices disagree")

    return {
        "wild_patch_len": wild_patch_len,
        "mutant_patch_len": mutant_patch_len,
        "wild_sequence_len": wild_sequence_len,
        "mutant_sequence_len": mutant_sequence_len,
    }


def convert_dataset(
    dataset: str,
    *,
    input_root: Path,
    neighbors: int,
    overwrite: bool,
    allow_sequence_fallback: bool,
) -> dict[str, Any]:
    input_path = input_root / dataset / "samples" / f"{dataset}_pythia_paired.pkl"
    output_path = (
        input_root
        / dataset
        / "samples"
        / f"{dataset}_pythia_patch{neighbors}_paired.pkl"
    )
    report_path = (
        input_root
        / dataset
        / "reports"
        / f"patch{neighbors}_report.json"
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing input pkl: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )

    with input_path.open("rb") as handle:
        items = pickle.load(handle)

    converted_items = []
    stats = {
        "patch_node_min": None,
        "patch_node_max": 0,
        "sequence_node_min": None,
        "sequence_node_max": 0,
    }
    for item in items:
        converted = convert_item(
            item,
            neighbors=neighbors,
            allow_sequence_fallback=allow_sequence_fallback,
        )
        lengths = validate_converted_item(converted, neighbors=neighbors)
        patch_len = max(lengths["wild_patch_len"], lengths["mutant_patch_len"])
        sequence_len = max(
            lengths["wild_sequence_len"],
            lengths["mutant_sequence_len"],
        )
        stats["patch_node_min"] = (
            patch_len
            if stats["patch_node_min"] is None
            else min(stats["patch_node_min"], patch_len)
        )
        stats["patch_node_max"] = max(stats["patch_node_max"], patch_len)
        stats["sequence_node_min"] = (
            sequence_len
            if stats["sequence_node_min"] is None
            else min(stats["sequence_node_min"], sequence_len)
        )
        stats["sequence_node_max"] = max(stats["sequence_node_max"], sequence_len)
        converted_items.append(converted)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(converted_items, handle, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "dataset": dataset,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "samples": len(converted_items),
        "neighbors": neighbors,
        "patch_max_nodes": neighbors + 1,
        "allow_sequence_fallback": allow_sequence_fallback,
        **stats,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired DDG pkl files with full sequence records and "
            "Pythia-style local structure patches"
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/ddg"),
        help="Root containing per-dataset ddg folders",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset folder names to convert",
    )
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-sequence-fallback", action="store_true")
    args = parser.parse_args()

    if args.neighbors < 1:
        raise ValueError("--neighbors must be positive")

    reports = []
    for dataset in args.datasets:
        report = convert_dataset(
            dataset.lower(),
            input_root=args.input_root,
            neighbors=args.neighbors,
            overwrite=args.overwrite,
            allow_sequence_fallback=args.allow_sequence_fallback,
        )
        reports.append(report)
        print(
            dataset,
            "samples",
            report["samples"],
            "patch_nodes",
            f'{report["patch_node_min"]}-{report["patch_node_max"]}',
            "sequence_nodes",
            f'{report["sequence_node_min"]}-{report["sequence_node_max"]}',
        )

    summary_path = args.input_root / f"patch{args.neighbors}_summary.json"
    summary_path.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
