from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa
from scipy.spatial import cKDTree

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional CLI dependency
    tqdm = None

from scripts.generate_structure_from_coordinate_cache import (
    AA3,
    AA_ORDER,
)


AA3_TO_INDEX = {AA3[aa]: index for index, aa in enumerate(AA_ORDER)}
PYTHIA_TOKEN_DIM = 22
PYTHIA_NODE_DIM = 28
PYTHIA_EDGE_DIM = 27
PYTHIA_K_NEIGHBORS = 32
PYTHIA_ATOMS = ["CA", "C", "N", "O", "CB"]
BACKBONE_ATOMS = ["N", "CA", "C", "O"]


def _read_manifest(path: Path) -> list[Path]:
    base_dir = path.parent
    paths: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", maxsplit=1)[0].strip()
        if not value:
            continue
        candidate = Path(value).expanduser()
        paths.append(candidate if candidate.is_absolute() else base_dir / candidate)
    return paths


def _residue_index(residue) -> int | None:
    return AA3_TO_INDEX.get(residue.get_resname().upper())


def _has_required_backbone(residue) -> bool:
    return all(residue.has_id(atom_name) for atom_name in BACKBONE_ATOMS)


def _chain_index(residue, chain_ids: dict[str, int]) -> int:
    chain_id = residue.get_parent().id
    if chain_id not in chain_ids:
        chain_ids[chain_id] = len(chain_ids)
    return chain_ids[chain_id]


def _virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    b = ca - n
    c_vec = c - ca
    a = np.cross(b, c_vec)
    return ca - 0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec


def _dihedral(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float:
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    norm = np.linalg.norm(b1)
    if norm <= 1e-6:
        return 0.0
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if v_norm <= 1e-6 or w_norm <= 1e-6:
        return 0.0
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def _bonded(c_atom: np.ndarray, n_atom: np.ndarray) -> bool:
    return bool(np.linalg.norm(c_atom - n_atom) < 2.0)


def _coordinate_payload(
    residues: list[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = len(residues)
    atom5 = np.zeros((length, len(PYTHIA_ATOMS), 3), dtype=np.float32)
    aa = np.zeros(length, dtype=np.int64)
    resseq = np.zeros(length, dtype=np.int64)
    chain = np.zeros(length, dtype=np.int64)
    chain_ids: dict[str, int] = {}

    for row, residue in enumerate(residues):
        index = _residue_index(residue)
        if index is None:
            raise ValueError(f"unsupported residue type: {residue.get_resname()}")
        aa[row] = index
        resseq[row] = int(residue.id[1])
        chain[row] = _chain_index(residue, chain_ids)
        n = residue["N"].get_coord()
        ca = residue["CA"].get_coord()
        c = residue["C"].get_coord()
        atom5[row, 0] = ca
        atom5[row, 1] = c
        atom5[row, 2] = n
        atom5[row, 3] = residue["O"].get_coord()
        if residue.has_id("CB"):
            atom5[row, 4] = residue["CB"].get_coord()
        else:
            atom5[row, 4] = _virtual_cb(n, ca, c)

    angles = _backbone_torsions(atom5, chain)
    return aa, atom5, chain, resseq, angles


def _backbone_torsions(atom5: np.ndarray, chain: np.ndarray) -> np.ndarray:
    length = atom5.shape[0]
    angles = np.zeros((length, 3), dtype=np.float32)
    ca = atom5[:, 0]
    c = atom5[:, 1]
    n = atom5[:, 2]

    for index in range(length):
        if (
            index > 0
            and chain[index - 1] == chain[index]
            and _bonded(c[index - 1], n[index])
        ):
            angles[index, 0] = _dihedral(
                c[index - 1], n[index], ca[index], c[index]
            )
        if (
            index + 1 < length
            and chain[index + 1] == chain[index]
            and _bonded(c[index], n[index + 1])
        ):
            angles[index, 1] = _dihedral(n[index], ca[index], c[index], n[index + 1])
            angles[index, 2] = _dihedral(
                ca[index], c[index], n[index + 1], ca[index + 1]
            )
    return angles


def pythia_node_features(aa: np.ndarray, angles: np.ndarray) -> np.ndarray:
    if angles.shape != (aa.shape[0], 3):
        raise ValueError("angles must have shape [num_residues, 3]")
    token_features = np.zeros((aa.shape[0], PYTHIA_TOKEN_DIM), dtype=np.float32)
    valid = (aa >= 0) & (aa < 20)
    token_features[valid, aa[valid]] = 1.0
    token_features[~valid, 20] = 1.0
    angle_features = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    return np.concatenate([token_features, angle_features], axis=1).astype(np.float32)


def _load_residues(
    path: Path,
    pdb_parser: PDBParser,
    cif_parser: MMCIFParser,
) -> list[Any]:
    parser = cif_parser if path.suffix.lower() == ".cif" else pdb_parser
    structure = parser.get_structure(path.stem, str(path))
    residues = [
        residue
        for residue in structure.get_residues()
        if (
            is_aa(residue, standard=True)
            and _residue_index(residue) is not None
            and _has_required_backbone(residue)
        )
    ]
    if len(residues) < 32:
        raise ValueError(
            f"too few standard residues with complete backbone: {len(residues)}"
        )
    return residues


def _pythia_edge_features(
    source: np.ndarray,
    target: np.ndarray,
    atom5: np.ndarray,
    chain: np.ndarray,
    resseq: np.ndarray,
    k_neighbors: int,
) -> np.ndarray:
    atom_dist = np.linalg.norm(
        atom5[source, :, None, :] - atom5[target, None, :, :],
        axis=-1,
    ).reshape(source.shape[0], -1)
    same_chain = (chain[source] == chain[target]).astype(np.float32)
    sequence_delta = np.zeros(source.shape[0], dtype=np.float32)
    if k_neighbors > 0:
        raw_delta = resseq[source] - resseq[target]
        sequence_delta = np.clip(raw_delta, -k_neighbors, k_neighbors).astype(
            np.float32
        ) / float(k_neighbors)
        sequence_delta *= same_chain
    return np.concatenate(
        [atom_dist, sequence_delta[:, None], same_chain[:, None]],
        axis=1,
    ).astype(np.float32)


def pythia_knn_graph(
    atom5: np.ndarray,
    chain: np.ndarray,
    resseq: np.ndarray,
    k_neighbors: int = PYTHIA_K_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")
    ca = atom5[:, 0]
    neighbor_count = min(k_neighbors, max(ca.shape[0] - 1, 0))
    if neighbor_count == 0:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, PYTHIA_EDGE_DIM), dtype=np.float32),
        )

    tree = cKDTree(ca)
    _, indices = tree.query(ca, k=neighbor_count + 1)
    if indices.ndim == 1:
        indices = indices[:, None]

    source_parts = []
    target_parts = []
    for target_index in range(ca.shape[0]):
        nearest = indices[target_index]
        nearest = nearest[nearest != target_index][:neighbor_count]
        if nearest.shape[0] < neighbor_count:
            distances = np.linalg.norm(ca - ca[target_index], axis=-1)
            distances[target_index] = np.inf
            nearest = np.argsort(distances)[:neighbor_count]
        source_parts.append(nearest.astype(np.int64))
        target_parts.append(
            np.full(neighbor_count, target_index, dtype=np.int64)
        )
    source = np.concatenate(source_parts)
    target = np.concatenate(target_parts)
    edges = np.stack([source, target], axis=0).astype(np.int64)
    edge_features = _pythia_edge_features(
        source,
        target,
        atom5,
        chain,
        resseq,
        k_neighbors=k_neighbors,
    )
    return edges, edge_features


def convert_pdb(
    path: Path,
    pdb_parser: PDBParser,
    cif_parser: MMCIFParser,
    k_neighbors: int,
) -> dict[str, Any]:
    residues = _load_residues(path, pdb_parser, cif_parser)
    aa, atom5, chain, resseq, angles = _coordinate_payload(residues)
    node_features = pythia_node_features(aa, angles)
    edges, edge_features = pythia_knn_graph(
        atom5,
        chain,
        resseq,
        k_neighbors=k_neighbors,
    )
    if edges.shape[1] == 0:
        raise ValueError("empty kNN graph")
    return {
        "name": path.name,
        "node_features": node_features,
        "edge_index": edges,
        "edge_features": edge_features,
        "ca_coords": atom5[:, 0].astype(np.float32),
        "mutation_pos": 0,
        "residue_type": aa.astype(np.int64),
        "pythia_atom_coords": atom5.astype(np.float32),
        "chain_index": chain.astype(np.int64),
        "resseq": resseq.astype(np.int64),
    }


def _flush_shard(
    records: list[dict[str, Any]],
    output_dir: Path,
    shard_index: int,
    prefix: str,
) -> Path:
    path = output_dir / f"{prefix}_{shard_index:04d}.pkl"
    with path.open("wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert structure PDB/mmCIF files to MutFlow-DDG graph records."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="cath_v4_3_s40_graphs")
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=PYTHIA_K_NEIGHBORS,
        help="Number of CA nearest neighbors per residue, matching Pythia.",
    )
    parser.add_argument("--report-json", default=None)
    parser.add_argument(
        "--progress",
        choices=("bar", "json", "none"),
        default="bar",
        help="Progress display mode. Default: tqdm progress bar.",
    )
    parser.add_argument("--progress-interval", type=int, default=1000)
    args = parser.parse_args()

    paths = _read_manifest(Path(args.manifest))
    if args.limit is not None:
        paths = paths[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_parser = PDBParser(QUIET=True)
    cif_parser = MMCIFParser(QUIET=True)

    shards: list[str] = []
    skipped: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    converted = 0
    progress_mode = args.progress
    if progress_mode == "bar" and tqdm is None:
        print(
            "tqdm is not installed; falling back to JSON progress. "
            "Install tqdm for a progress bar.",
            flush=True,
        )
        progress_mode = "json"
    progress_bar = (
        tqdm(total=len(paths), desc="Converting structures", unit="structure")
        if progress_mode == "bar"
        else None
    )

    for index, path in enumerate(paths, start=1):
        try:
            records.append(
                convert_pdb(
                    path,
                    pdb_parser,
                    cif_parser,
                    k_neighbors=args.k_neighbors,
                )
            )
            converted += 1
        except Exception as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
        if records and len(records) >= args.shard_size:
            shards.append(
                str(_flush_shard(records, output_dir, len(shards), args.prefix))
            )
            records = []
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(
                converted=converted,
                skipped=len(skipped),
                shards=len(shards),
            )
        elif (
            progress_mode == "json"
            and args.progress_interval > 0
            and index % args.progress_interval == 0
        ):
            print(
                json.dumps(
                    {
                        "seen": index,
                        "converted": converted,
                        "skipped": len(skipped),
                    }
                ),
                flush=True,
            )

    if progress_bar is not None:
        progress_bar.close()
    if records:
        shards.append(str(_flush_shard(records, output_dir, len(shards), args.prefix)))

    shard_manifest = output_dir / f"{args.prefix}_shards.txt"
    shard_manifest.write_text("\n".join(shards) + "\n", encoding="utf-8")
    report = {
        "manifest": str(args.manifest),
        "output_dir": str(output_dir),
        "total_paths": len(paths),
        "converted": converted,
        "skipped": len(skipped),
        "shards": shards,
        "shard_manifest": str(shard_manifest),
        "skipped_examples": skipped[:20],
    }
    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
