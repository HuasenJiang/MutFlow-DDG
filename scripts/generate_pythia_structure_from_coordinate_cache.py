from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from mutflow_ddg.data.identity import (
    baseline_rows,
    match_items_to_baseline,
    parse_mutation,
    parse_record_name,
)
from scripts.convert_cath_pdb_to_graphs import (
    PYTHIA_EDGE_DIM,
    PYTHIA_K_NEIGHBORS,
    PYTHIA_NODE_DIM,
    _backbone_torsions,
    _coordinate_payload,
    _has_required_backbone,
    _residue_index,
    _virtual_cb,
    pythia_knn_graph,
    pythia_node_features,
)
from scripts.generate_structure_from_coordinate_cache import (
    AA_ORDER,
    clean_name,
    load_coordinate,
    resolve_coordinate_path,
    valid_atom_coords,
)

BIOPYTHON_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
BIOPYTHON_TO_MUTFLOW_AA = np.asarray(
    [AA_ORDER.index(amino_acid) for amino_acid in BIOPYTHON_AA_ORDER],
    dtype=np.int64,
)


def _as_array(value: Any, dtype: np.dtype) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _residue_type_from_record(record: dict[str, Any]) -> np.ndarray | None:
    node_features = _as_array(record["node_features"], np.float32)
    if node_features.ndim != 2 or node_features.shape[1] < 20:
        return None
    residue_tokens = node_features[:, :20]
    valid = residue_tokens.max(axis=1) > 0
    if not valid.any():
        return None
    result = residue_tokens.argmax(axis=1).astype(np.int64)
    result[~valid] = -1
    return result


def _residue_type_from_coordinate(
    coord: dict[str, Any],
    length: int,
) -> np.ndarray | None:
    value = coord.get("aa")
    if value is None:
        return None
    aa = _as_array(value, np.int64)
    if aa.shape != (length,):
        return None
    if ((aa < 0) | (aa >= BIOPYTHON_TO_MUTFLOW_AA.shape[0])).any():
        return None
    return BIOPYTHON_TO_MUTFLOW_AA[aa]


def _residue_type(
    record: dict[str, Any],
    coord: dict[str, Any],
    length: int,
) -> np.ndarray:
    aa = _residue_type_from_record(record)
    if aa is None:
        aa = _residue_type_from_coordinate(coord, length)
    if aa is None:
        raise ValueError("cannot infer residue types from record or coordinate")
    if aa.shape[0] != length:
        raise ValueError(
            f"residue type length mismatch: residue_type={aa.shape[0]}, "
            f"coordinate={length}"
        )
    return aa.astype(np.int64)


def _resseq(coord: dict[str, Any], length: int) -> np.ndarray:
    value = coord.get("resseq")
    if value is None:
        return np.arange(length, dtype=np.int64)
    resseq = _as_array(value, np.int64)
    if resseq.shape != (length,):
        raise ValueError(
            f"resseq length mismatch: resseq={resseq.shape}, length={length}"
        )
    return resseq


def _chain_index(coord: dict[str, Any], length: int) -> np.ndarray:
    value = coord.get("chain_index")
    if value is None:
        return np.zeros(length, dtype=np.int64)
    chain = _as_array(value, np.int64)
    if chain.shape != (length,):
        raise ValueError(
            f"chain_index length mismatch: chain_index={chain.shape}, "
            f"length={length}"
        )
    return chain


def _atom5_from_coordinate(coord: dict[str, Any]) -> np.ndarray:
    pos14, mask = valid_atom_coords(coord)
    if pos14.ndim != 3 or pos14.shape[1] < 4 or pos14.shape[2] != 3:
        raise ValueError(
            "coordinate pos14 must have shape [num_residues, >=4, 3]"
        )
    length = pos14.shape[0]
    backbone_present = mask[:, :4].all(axis=1)
    if not backbone_present.all():
        missing = np.nonzero(~backbone_present)[0][:5].tolist()
        raise ValueError(f"missing N/CA/C/O backbone atoms at residues {missing}")

    atom5 = np.zeros((length, 5, 3), dtype=np.float32)
    atom5[:, 0] = pos14[:, 1]  # CA
    atom5[:, 1] = pos14[:, 2]  # C
    atom5[:, 2] = pos14[:, 0]  # N
    atom5[:, 3] = pos14[:, 3]  # O
    if pos14.shape[1] > 4:
        cb_present = mask[:, 4]
        atom5[cb_present, 4] = pos14[cb_present, 4]
        atom5[~cb_present, 4] = _virtual_cb(
            pos14[~cb_present, 0],
            pos14[~cb_present, 1],
            pos14[~cb_present, 2],
        )
    else:
        atom5[:, 4] = _virtual_cb(pos14[:, 0], pos14[:, 1], pos14[:, 2])
    return atom5


def convert_record(
    record: dict[str, Any],
    coord: dict[str, Any],
    k_neighbors: int = PYTHIA_K_NEIGHBORS,
) -> dict[str, Any]:
    atom5 = _atom5_from_coordinate(coord)
    length = atom5.shape[0]
    original_node_features = _as_array(record["node_features"], np.float32)
    if original_node_features.ndim != 2:
        raise ValueError("record node_features must have shape [num_residues, D]")
    if original_node_features.shape[0] != length:
        raise ValueError(
            "length mismatch: "
            f"node_features={original_node_features.shape[0]}, coordinate={length}"
        )

    aa = _residue_type(record, coord, length)
    chain = _chain_index(coord, length)
    resseq = _resseq(coord, length)
    angles = _backbone_torsions(atom5, chain)
    node_features = pythia_node_features(aa, angles)
    edge_index, edge_features = pythia_knn_graph(
        atom5,
        chain,
        resseq,
        k_neighbors=k_neighbors,
    )
    if node_features.shape[1] != PYTHIA_NODE_DIM:
        raise ValueError(f"unexpected Pythia node dimension: {node_features.shape}")
    if edge_features.shape[1] != PYTHIA_EDGE_DIM:
        raise ValueError(f"unexpected Pythia edge dimension: {edge_features.shape}")

    converted = dict(record)
    converted["node_features"] = node_features
    converted["edge_index"] = edge_index
    converted["edge_features"] = edge_features
    converted["mutation_pos"] = int(record.get("mutation_pos", 0))
    converted["residue_type"] = aa
    converted["ca_coords"] = atom5[:, 0].astype(np.float32)
    converted["pythia_atom_coords"] = atom5.astype(np.float32)
    converted["chain_index"] = chain.astype(np.int64)
    converted["resseq"] = resseq.astype(np.int64)
    return converted


def _pdb_path(pdb_dir: Path, record_name: str) -> Path | None:
    name = clean_name(record_name)
    candidates = [pdb_dir / f"{name}.pdb", pdb_dir / name]
    return next((path for path in candidates if path.is_file()), None)


def _target_chain_residues(path: Path, chain_id: str) -> list[Any]:
    model = PDBParser(QUIET=True).get_structure(path.stem, str(path))[0]
    if not model.has_id(chain_id):
        available = ",".join(chain.id for chain in model)
        raise ValueError(
            f"target chain {chain_id!r} absent from PDB; available={available}"
        )
    residues = [
        residue
        for residue in model[chain_id]
        if (
            is_aa(residue, standard=True)
            and _residue_index(residue) is not None
            and _has_required_backbone(residue)
        )
    ]
    if len(residues) < 2:
        raise ValueError(
            f"target chain {chain_id!r} has too few complete residues"
        )
    return residues


def _mutation_index_from_residues(
    residues: list[Any],
    mutation: str,
    *,
    mutant: bool,
) -> int:
    wild_aa, position, mutant_aa = parse_mutation(mutation)
    expected = mutant_aa if mutant else wild_aa
    matches = [
        index
        for index, residue in enumerate(residues)
        if f"{int(residue.id[1])}{str(residue.id[2]).strip()}" == position
    ]
    if len(matches) != 1:
        raise ValueError(
            f"mutation residue {position!r} matched {len(matches)} target-chain "
            "residues"
        )
    index = matches[0]
    residue_aa_index = _residue_index(residues[index])
    observed = AA_ORDER[residue_aa_index] if residue_aa_index is not None else "X"
    if observed != expected:
        raise ValueError(
            f"mutation identity mismatch at {position}: expected {expected}, "
            f"PDB has {observed}"
        )
    return index


def convert_record_from_target_chain_pdb(
    record: dict[str, Any],
    path: Path,
    *,
    chain_id: str,
    mutation: str,
    mutant: bool,
    k_neighbors: int,
) -> dict[str, Any]:
    """Rebuild a graph from the explicitly mutated chain, never the first chain."""

    residues = _target_chain_residues(path, chain_id)
    aa, atom5, chain, resseq, angles = _coordinate_payload(residues)
    mutation_pos = _mutation_index_from_residues(
        residues,
        mutation,
        mutant=mutant,
    )
    node_features = pythia_node_features(aa, angles)
    edge_index, edge_features = pythia_knn_graph(
        atom5,
        chain,
        resseq,
        k_neighbors=k_neighbors,
    )
    converted = dict(record)
    converted.update(
        {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "mutation_pos": mutation_pos,
            "residue_type": aa,
            "ca_coords": atom5[:, 0].astype(np.float32),
            "pythia_atom_coords": atom5.astype(np.float32),
            "chain_index": chain.astype(np.int64),
            "resseq": resseq.astype(np.int64),
        }
    )
    return converted


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [{name: row.get(name, "") for name in fieldnames} for row in rows]
        )


def convert_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_pkl = Path(args.input_pkl)
    output_pkl = Path(args.output_pkl)
    cache_dir = Path(args.coordinate_cache_dir)

    with input_pkl.open("rb") as handle:
        data = pickle.load(handle)
    if args.limit is not None:
        data = data[: args.limit]
    reference_rows = (
        baseline_rows(args.baseline_csv, args.dataset_name)
        if args.baseline_csv
        else []
    )
    item_rows = (
        match_items_to_baseline(data, reference_rows)
        if reference_rows
        else [None] * len(data)
    )

    converted = []
    skipped = []
    manifest_rows = []
    for index, item in enumerate(data):
        wild_name = clean_name(item.get("wild_type_name", ""))
        mutant_name = clean_name(item.get("mutant_name", ""))
        mutant_parts = parse_record_name(mutant_name)
        baseline_row = item_rows[index]
        if baseline_row is None:
            baseline_row = {
                "dataset": args.dataset_name.lower(),
                "source_row_index": "",
                "protein_id": mutant_parts["protein_id"],
                "chain": mutant_parts["chain"],
                "mutation": mutant_parts["mutation"],
                "ddg": float(item["ddg"]),
                "mutation_id": (
                    f"{args.dataset_name.lower()}:input:{index:06d}:"
                    f"{mutant_parts['protein_id']}:{mutant_parts['chain']}:"
                    f"{mutant_parts['mutation']}"
                ),
            }
        wild_coord_path = resolve_coordinate_path(cache_dir, wild_name, args.chain)
        mutant_coord_path = resolve_coordinate_path(cache_dir, mutant_name, args.chain)
        recovery_source = "coordinate_cache"
        try:
            coordinate_error = None
            try:
                if wild_coord_path is None or mutant_coord_path is None:
                    raise FileNotFoundError("missing_coordinate")
                wild = convert_record(
                    item["wild_type"],
                    load_coordinate(wild_coord_path),
                    k_neighbors=args.k_neighbors,
                )
                mutant = convert_record(
                    item["mutant"],
                    load_coordinate(mutant_coord_path),
                    k_neighbors=args.k_neighbors,
                )
            except Exception as exc:
                coordinate_error = str(exc)
                if args.pdb_dir is None:
                    raise
                pdb_dir = Path(args.pdb_dir)
                wild_pdb = _pdb_path(pdb_dir, wild_name)
                mutant_pdb = _pdb_path(pdb_dir, mutant_name)
                if wild_pdb is None or mutant_pdb is None:
                    raise FileNotFoundError(
                        f"{coordinate_error}; target-chain PDB fallback missing"
                    ) from exc
                chain_id = mutant_parts["chain"]
                mutation = mutant_parts["mutation"]
                wild = convert_record_from_target_chain_pdb(
                    item["wild_type"],
                    wild_pdb,
                    chain_id=chain_id,
                    mutation=mutation,
                    mutant=False,
                    k_neighbors=args.k_neighbors,
                )
                mutant = convert_record_from_target_chain_pdb(
                    item["mutant"],
                    mutant_pdb,
                    chain_id=chain_id,
                    mutation=mutation,
                    mutant=True,
                    k_neighbors=args.k_neighbors,
                )
                recovery_source = "target_chain_pdb"
            if wild["node_features"].shape[0] != mutant["node_features"].shape[0]:
                raise ValueError(
                    "WT/MUT length mismatch after conversion: "
                    f"{wild['node_features'].shape[0]} vs "
                    f"{mutant['node_features'].shape[0]}"
                )
            if wild["edge_index"].shape[1] == 0 or mutant["edge_index"].shape[1] == 0:
                raise ValueError("empty Pythia kNN graph")

            converted_item = dict(item)
            converted_item["wild_type"] = wild
            converted_item["mutant"] = mutant
            if baseline_row is not None:
                converted_item["mutation_id"] = baseline_row["mutation_id"]
                converted_item["source_row_index"] = baseline_row[
                    "source_row_index"
                ]
            converted.append(converted_item)
            manifest_rows.append(
                {
                    **(baseline_row or {}),
                    "input_index": index,
                    "status": "success",
                    "reason": "",
                    "recovery_source": recovery_source,
                    "wild_type_name": wild_name,
                    "mutant_name": mutant_name,
                    "ddg": float(item["ddg"]),
                }
            )
        except Exception as exc:
            failure = {
                "index": index,
                "wild_type_name": wild_name,
                "mutant_name": mutant_name,
                "reason": str(exc),
                "wild_coordinate": (
                    str(wild_coord_path) if wild_coord_path is not None else None
                ),
                "mutant_coordinate": (
                    str(mutant_coord_path)
                    if mutant_coord_path is not None
                    else None
                ),
            }
            skipped.append(failure)
            manifest_rows.append(
                {
                    **(baseline_row or {}),
                    "input_index": index,
                    "status": "failed",
                    "reason": str(exc),
                    "recovery_source": "",
                    "wild_type_name": wild_name,
                    "mutant_name": mutant_name,
                    "ddg": float(item["ddg"]),
                }
            )

    if reference_rows:
        represented = {
            row["mutation_id"]
            for row in manifest_rows
            if row.get("mutation_id")
        }
        for row in reference_rows:
            if row["mutation_id"] not in represented:
                manifest_rows.append(
                    {
                        **row,
                        "input_index": "",
                        "status": "failed",
                        "reason": "missing_from_input_pickle",
                        "recovery_source": "",
                        "wild_type_name": "",
                        "mutant_name": "",
                    }
                )

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as handle:
        pickle.dump(converted, handle, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "input_pkl": str(input_pkl),
        "output_pkl": str(output_pkl),
        "coordinate_cache_dir": str(cache_dir),
        "k_neighbors": args.k_neighbors,
        "total_samples": len(data),
        "converted_samples": len(converted),
        "skipped_samples": len(skipped),
        "baseline_samples": len(reference_rows) or None,
        "baseline_missing_samples": (
            sum(row["status"] == "failed" for row in manifest_rows)
            if reference_rows
            else None
        ),
        "recovered_from_target_chain_pdb": sum(
            row.get("recovery_source") == "target_chain_pdb"
            for row in manifest_rows
        ),
        "skipped": skipped,
    }
    if args.manifest_dir:
        manifest_dir = Path(args.manifest_dir)
        _write_manifest(manifest_dir / "all_samples.csv", manifest_rows)
        _write_manifest(
            manifest_dir / "success_samples.csv",
            [row for row in manifest_rows if row["status"] == "success"],
        )
        _write_manifest(
            manifest_dir / "failed_samples.csv",
            [row for row in manifest_rows if row["status"] == "failed"],
        )
    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.fail_on_skipped and skipped:
        examples = "; ".join(item["reason"] for item in skipped[:3])
        raise RuntimeError(
            f"{len(skipped)} samples were skipped during conversion: {examples}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert paired DDG graph pickles to Pythia-style 28-D node and "
            "27-D edge structural features from coordinate.pt caches."
        )
    )
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--coordinate-cache-dir", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--baseline-csv", default=None)
    parser.add_argument(
        "--pdb-dir",
        default=None,
        help="Fallback directory used to rebuild the explicitly mutated chain.",
    )
    parser.add_argument(
        "--manifest-dir",
        default=None,
        help="Write all/success/failed mutation-ID manifests here.",
    )
    parser.add_argument("--chain", default="A")
    parser.add_argument("--k-neighbors", type=int, default=PYTHIA_K_NEIGHBORS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-on-skipped", action="store_true")
    args = parser.parse_args()

    report = convert_dataset(args)
    print(json.dumps({k: v for k, v in report.items() if k != "skipped"}, indent=2))
    if report["skipped_samples"]:
        print("first_skipped", json.dumps(report["skipped"][:5], indent=2))


if __name__ == "__main__":
    main()
