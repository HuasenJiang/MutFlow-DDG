from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


AA_ORDER = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
AA3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

KD_HYDROPHOBICITY = {
    "ALA": 0.700,
    "CYS": 0.778,
    "ASP": 0.111,
    "GLU": 0.111,
    "PHE": 0.811,
    "GLY": 0.456,
    "HIS": 0.144,
    "ILE": 1.000,
    "LYS": 0.067,
    "LEU": 0.922,
    "MET": 0.711,
    "ASN": 0.111,
    "PRO": 0.322,
    "GLN": 0.111,
    "ARG": 0.000,
    "SER": 0.411,
    "THR": 0.422,
    "VAL": 0.967,
    "TRP": 0.400,
    "TYR": 0.356,
}
RESIDUE_VOLUMES = {
    "ALA": 0.168,
    "CYS": 0.285,
    "ASP": 0.301,
    "GLU": 0.461,
    "PHE": 0.764,
    "GLY": 0.001,
    "HIS": 0.548,
    "ILE": 0.628,
    "LYS": 0.639,
    "LEU": 0.628,
    "MET": 0.605,
    "ASN": 0.318,
    "PRO": 0.310,
    "GLN": 0.493,
    "ARG": 0.667,
    "SER": 0.171,
    "THR": 0.330,
    "VAL": 0.471,
    "TRP": 0.987,
    "TYR": 0.786,
}
RESIDUE_POLARITY = {
    "ALA": 0.000,
    "CYS": 0.028,
    "ASP": 0.783,
    "GLU": 0.783,
    "PHE": 0.007,
    "GLY": 0.000,
    "HIS": 0.068,
    "ILE": 0.003,
    "LYS": 0.952,
    "LEU": 0.003,
    "MET": 0.028,
    "ASN": 0.065,
    "PRO": 0.030,
    "GLN": 0.068,
    "ARG": 1.000,
    "SER": 0.032,
    "THR": 0.032,
    "VAL": 0.003,
    "TRP": 0.040,
    "TYR": 0.031,
}
RESIDUE_PKA = {
    "ALA": 0.000,
    "CYS": 0.541,
    "ASP": 0.046,
    "GLU": 0.072,
    "PHE": 0.000,
    "GLY": 0.000,
    "HIS": 0.282,
    "ILE": 0.000,
    "LYS": 0.797,
    "LEU": 0.000,
    "MET": 0.000,
    "ASN": 0.000,
    "PRO": 0.000,
    "GLN": 0.000,
    "ARG": 0.956,
    "SER": 0.000,
    "THR": 0.000,
    "VAL": 0.000,
    "TRP": 0.000,
    "TYR": 0.773,
}
RESIDUE_CHARGE = {
    "ALA": 0.0,
    "CYS": 0.0,
    "ASP": -1.0,
    "GLU": -1.0,
    "PHE": 0.0,
    "GLY": 0.0,
    "HIS": 0.1,
    "ILE": 0.0,
    "LYS": 1.0,
    "LEU": 0.0,
    "MET": 0.0,
    "ASN": 0.0,
    "PRO": 0.0,
    "GLN": 0.0,
    "ARG": 1.0,
    "SER": 0.0,
    "THR": 0.0,
    "VAL": 0.0,
    "TRP": 0.0,
    "TYR": 0.0,
}
FLEXIBILITY = {
    "ALA": 0.360,
    "CYS": 0.350,
    "ASP": 0.510,
    "GLU": 0.500,
    "PHE": 0.310,
    "GLY": 0.540,
    "HIS": 0.320,
    "ILE": 0.460,
    "LYS": 0.470,
    "LEU": 0.370,
    "MET": 0.300,
    "ASN": 0.460,
    "PRO": 0.510,
    "GLN": 0.490,
    "ARG": 0.530,
    "SER": 0.510,
    "THR": 0.440,
    "VAL": 0.390,
    "TRP": 0.310,
    "TYR": 0.420,
}
BETA = {
    "ALA": 0.83,
    "CYS": 1.19,
    "ASP": 0.54,
    "GLU": 0.37,
    "PHE": 1.38,
    "GLY": 0.75,
    "HIS": 0.87,
    "ILE": 1.60,
    "LYS": 0.74,
    "LEU": 1.30,
    "MET": 1.05,
    "ASN": 0.89,
    "PRO": 0.55,
    "GLN": 1.10,
    "ARG": 0.93,
    "SER": 0.75,
    "THR": 1.19,
    "VAL": 1.70,
    "TRP": 1.37,
    "TYR": 1.47,
}
TURN = {
    "ALA": 0.66,
    "CYS": 1.19,
    "ASP": 1.46,
    "GLU": 0.74,
    "PHE": 0.60,
    "GLY": 1.56,
    "HIS": 0.95,
    "ILE": 0.47,
    "LYS": 1.01,
    "LEU": 0.59,
    "MET": 0.60,
    "ASN": 1.56,
    "PRO": 1.52,
    "GLN": 0.98,
    "ARG": 0.95,
    "SER": 1.43,
    "THR": 0.96,
    "VAL": 0.50,
    "TRP": 0.96,
    "TYR": 1.14,
}
EXPOSURE = {
    "ALA": 0.48,
    "CYS": 0.32,
    "ASP": 0.81,
    "GLU": 0.93,
    "PHE": 0.23,
    "GLY": 0.51,
    "HIS": 0.66,
    "ILE": 0.34,
    "LYS": 0.97,
    "LEU": 0.40,
    "MET": 0.38,
    "ASN": 0.82,
    "PRO": 0.78,
    "GLN": 0.81,
    "ARG": 0.84,
    "SER": 0.70,
    "THR": 0.71,
    "VAL": 0.40,
    "TRP": 0.31,
    "TYR": 0.42,
}
ALPHA = {
    "ALA": 1.42,
    "CYS": 0.70,
    "ASP": 1.01,
    "GLU": 1.51,
    "PHE": 1.13,
    "GLY": 0.57,
    "HIS": 1.00,
    "ILE": 1.08,
    "LYS": 1.16,
    "LEU": 1.21,
    "MET": 1.45,
    "ASN": 0.67,
    "PRO": 0.57,
    "GLN": 1.11,
    "ARG": 0.98,
    "SER": 0.77,
    "THR": 0.83,
    "VAL": 1.06,
    "TRP": 1.08,
    "TYR": 0.69,
}
MAX_ACC = {
    "ALA": 121.0,
    "ARG": 265.0,
    "ASN": 187.0,
    "ASP": 187.0,
    "CYS": 148.0,
    "GLN": 214.0,
    "GLU": 214.0,
    "GLY": 97.0,
    "HIS": 216.0,
    "ILE": 195.0,
    "LEU": 191.0,
    "LYS": 230.0,
    "MET": 203.0,
    "PHE": 228.0,
    "PRO": 154.0,
    "SER": 143.0,
    "THR": 163.0,
    "TRP": 264.0,
    "TYR": 255.0,
    "VAL": 165.0,
}


def clean_name(name: Any) -> str:
    value = Path(str(name)).name
    return value[:-4] if value.lower().endswith(".pdb") else value


def coordinate_candidates(name: str, chain: str) -> list[str]:
    candidates = [name]
    candidates.append(
        name.replace("_wild_type", f"_{chain}_wild_type").replace(
            "_mutant_", f"_{chain}_mutant_"
        )
    )
    match = re.match(r"^([0-9][A-Za-z0-9]{3})([A-Za-z])_(wild_type|mutant_.+)$", name)
    if match:
        candidates.append(
            f"{match.group(1).upper()}_{match.group(2).upper()}_{match.group(3)}"
        )
    return list(dict.fromkeys(candidates))


def resolve_coordinate_path(cache_dir: Path, name: str, chain: str) -> Path | None:
    for candidate in coordinate_candidates(name, chain):
        path = cache_dir / candidate / "coordinate.pt"
        if path.is_file():
            return path
    return None


def load_coordinate(path: Path) -> dict[str, np.ndarray]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return {
        key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
        for key, value in raw.items()
    }


def residue_names(node_features: np.ndarray) -> list[str]:
    one_hot = node_features[:, :20]
    indices = one_hot.argmax(axis=1)
    valid = one_hot.max(axis=1) > 0
    names = []
    for index, is_valid in zip(indices, valid):
        names.append(AA3[AA_ORDER[int(index)]] if is_valid else "ALA")
    return names


def sequence_properties(node_features: np.ndarray) -> np.ndarray:
    props = []
    for resname in residue_names(node_features):
        coil = 1.0 - (ALPHA[resname] + BETA[resname]) / 2.0
        props.append(
            [
                KD_HYDROPHOBICITY[resname],
                RESIDUE_VOLUMES[resname],
                RESIDUE_POLARITY[resname],
                RESIDUE_PKA[resname],
                RESIDUE_CHARGE[resname],
                FLEXIBILITY[resname],
                BETA[resname],
                TURN[resname],
                EXPOSURE[resname],
                ALPHA[resname],
                BETA[resname],
                coil,
            ]
        )
    return np.asarray(props, dtype=np.float32)


def valid_atom_coords(coord: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pos14 = np.asarray(coord["pos14"], dtype=np.float32)
    mask = np.asarray(coord["pos14_mask"], dtype=bool)
    if mask.ndim == 3:
        mask = mask.all(axis=-1)
    finite = np.isfinite(pos14).all(axis=-1)
    mask = mask & finite
    return pos14, mask


def ca_coords(coord: dict[str, np.ndarray]) -> np.ndarray:
    pos14, mask = valid_atom_coords(coord)
    if pos14.shape[1] < 2:
        raise ValueError("coordinate cache does not contain CA atoms")
    ca = pos14[:, 1, :]
    ca_mask = mask[:, 1]
    if not ca_mask.all():
        raise ValueError("coordinate cache has missing CA atoms")
    return ca


def positional_encoding(distance: np.ndarray, dimension: int = 16) -> np.ndarray:
    parts = []
    for i in range(dimension):
        denom = 10000 ** ((2 * i) / dimension)
        values = np.sin(distance / denom) if i % 2 == 0 else np.cos(distance / denom)
        parts.append(values)
    return np.stack(parts, axis=1).astype(np.float32)


def edge_graph(ca: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    diff = ca[:, None, :] - ca[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    mask = np.triu((dist < cutoff) & (dist > 0), k=1)
    src, dst = np.nonzero(mask)
    if src.size == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 16), dtype=np.float32)
    edges = np.stack([src, dst], axis=0).astype(np.int64)
    edge_features = positional_encoding(dist[src, dst])
    return edges, edge_features


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)


def atom_depth(atom_coords: np.ndarray, all_atoms: np.ndarray) -> float:
    if atom_coords.shape[0] == 0 or all_atoms.shape[0] <= 1:
        return 0.0
    dists = np.linalg.norm(atom_coords[:, None, :] - all_atoms[None, :, :], axis=-1)
    dists[dists <= 1e-6] = np.inf
    nearest = np.min(dists, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return 0.0
    return float(np.clip(nearest.mean() / 10.0, 0.0, 1.0))


def structure_features(
    coord: dict[str, np.ndarray],
    node_features: np.ndarray,
    contact_cutoff: float,
    local_radius: float,
) -> np.ndarray:
    pos14, atom_mask = valid_atom_coords(coord)
    ca = ca_coords(coord)
    ca_dist = pairwise_distances(ca)
    names = residue_names(node_features)
    length = ca.shape[0]

    features = np.zeros((length, 14), dtype=np.float32)
    features[:, 2] = 1.0

    contact_mask_8 = (ca_dist < contact_cutoff) & (ca_dist > 0)
    contact_mask_10 = (ca_dist < local_radius) & (ca_dist > 0)
    denom = max(length - 1, 1)

    all_atoms = pos14[atom_mask]
    atom_residue_index = np.repeat(np.arange(length), atom_mask.sum(axis=1))
    volume = (4.0 / 3.0) * np.pi * (local_radius**3)

    for i in range(length):
        contact_density = float(contact_mask_10[i].sum() / denom)
        rsa = float(np.clip(1.0 - contact_density / 0.30, 0.0, 1.0))
        asa_norm = min((rsa * MAX_ACC.get(names[i], 200.0)) / 1000.0, 1.0)

        local_atom_count = 0
        if all_atoms.size:
            local_atom_count = int(
                (np.linalg.norm(all_atoms - ca[i], axis=1) <= local_radius).sum()
            )
        structure_density = min((local_atom_count / volume) / 0.1, 1.0)

        contacts = np.nonzero(contact_mask_8[i])[0]
        if contacts.size:
            relative_contact_order = float(np.abs(contacts - i).sum() / (contacts.size * length))
        else:
            relative_contact_order = 0.0

        residue_atoms = pos14[i][atom_mask[i]]
        backbone_atoms = pos14[i, :4][atom_mask[i, :4]]
        sidechain_atoms = pos14[i, 4:][atom_mask[i, 4:]]
        mc_depth = atom_depth(backbone_atoms, all_atoms)
        sc_depth = atom_depth(sidechain_atoms if sidechain_atoms.size else residue_atoms, all_atoms)

        backbone_hbond = 0.0
        sidechain_hbond = 0.0
        if atom_mask[i, 0] and atom_mask[i, 3]:
            own_n = pos14[i, 0]
            own_o = pos14[i, 3]
            other_residues = np.arange(length) != i
            other_n = pos14[other_residues, 0][atom_mask[other_residues, 0]]
            other_o = pos14[other_residues, 3][atom_mask[other_residues, 3]]
            if other_o.size:
                backbone_hbond += float((np.linalg.norm(other_o - own_n, axis=1) < 3.5).sum())
            if other_n.size:
                backbone_hbond += float((np.linalg.norm(other_n - own_o, axis=1) < 3.5).sum())
            backbone_hbond = min(backbone_hbond / 4.0, 1.0)
        if sidechain_atoms.size and all_atoms.size:
            other_atoms = all_atoms[atom_residue_index != i]
            if other_atoms.size:
                nearest = np.min(
                    np.linalg.norm(sidechain_atoms[:, None, :] - other_atoms[None, :, :], axis=-1),
                    axis=1,
                )
                sidechain_hbond = min(float((nearest < 3.5).sum()) / 8.0, 1.0)

        features[i, 3:5] = [rsa, asa_norm]
        features[i, 5] = 0.0
        features[i, 6:14] = [
            contact_density,
            backbone_hbond,
            sidechain_hbond,
            0.0,
            structure_density,
            relative_contact_order,
            mc_depth,
            sc_depth,
        ]

    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)


def convert_record(
    record: dict[str, Any],
    coord: dict[str, np.ndarray],
    contact_cutoff: float,
    local_radius: float,
) -> dict[str, Any]:
    base = np.asarray(record["node_features"], dtype=np.float32)
    if base.ndim != 2 or base.shape[1] < 41:
        raise ValueError(f"expected at least 41 node features, got {base.shape}")
    base = base[:, :41]
    if ca_coords(coord).shape[0] != base.shape[0]:
        raise ValueError(
            f"length mismatch: node_features={base.shape[0]}, coordinate={ca_coords(coord).shape[0]}"
        )
    seq_props = sequence_properties(base)
    struct = structure_features(coord, base, contact_cutoff, local_radius)
    node_features = np.concatenate([base, seq_props, struct], axis=1).astype(np.float32)
    edges, edge_features = edge_graph(ca_coords(coord), contact_cutoff)
    return {
        "node_features": node_features,
        "edge_index": edges,
        "edge_features": edge_features,
        "mutation_pos": int(record.get("mutation_pos", 0)),
    }


def convert_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_pkl = Path(args.input_pkl)
    output_pkl = Path(args.output_pkl)
    cache_dir = Path(args.coordinate_cache_dir)

    with input_pkl.open("rb") as handle:
        data = pickle.load(handle)

    converted = []
    skipped = []
    for index, item in enumerate(data):
        wild_name = clean_name(item.get("wild_type_name", ""))
        mutant_name = clean_name(item.get("mutant_name", ""))
        wild_coord_path = resolve_coordinate_path(cache_dir, wild_name, args.chain)
        mutant_coord_path = resolve_coordinate_path(cache_dir, mutant_name, args.chain)
        if wild_coord_path is None or mutant_coord_path is None:
            skipped.append(
                {
                    "index": index,
                    "wild_type_name": wild_name,
                    "mutant_name": mutant_name,
                    "reason": "missing_coordinate",
                    "wild_coordinate": str(wild_coord_path) if wild_coord_path else None,
                    "mutant_coordinate": str(mutant_coord_path) if mutant_coord_path else None,
                }
            )
            continue

        try:
            wild_coord = load_coordinate(wild_coord_path)
            mutant_coord = load_coordinate(mutant_coord_path)
            wild = convert_record(
                item["wild_type"], wild_coord, args.contact_cutoff, args.local_radius
            )
            mutant = convert_record(
                item["mutant"], mutant_coord, args.contact_cutoff, args.local_radius
            )
            if wild["node_features"].shape[0] != mutant["node_features"].shape[0]:
                raise ValueError(
                    "WT/MUT length mismatch after conversion: "
                    f"{wild['node_features'].shape[0]} vs {mutant['node_features'].shape[0]}"
                )
            if wild["edge_index"].shape[1] == 0 or mutant["edge_index"].shape[1] == 0:
                raise ValueError("empty contact graph")

            converted_item = dict(item)
            converted_item["wild_type"] = wild
            converted_item["mutant"] = mutant
            converted.append(converted_item)
        except Exception as exc:
            skipped.append(
                {
                    "index": index,
                    "wild_type_name": wild_name,
                    "mutant_name": mutant_name,
                    "reason": str(exc),
                    "wild_coordinate": str(wild_coord_path),
                    "mutant_coordinate": str(mutant_coord_path),
                }
            )

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as handle:
        pickle.dump(converted, handle)

    report = {
        "input_pkl": str(input_pkl),
        "output_pkl": str(output_pkl),
        "coordinate_cache_dir": str(cache_dir),
        "total_samples": len(data),
        "converted_samples": len(converted),
        "skipped_samples": len(skipped),
        "skipped": skipped,
    }
    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MutFlow-DDG structure-branch pkl files from coordinate.pt caches."
    )
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--coordinate-cache-dir", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--local-radius", type=float, default=10.0)
    args = parser.parse_args()

    report = convert_dataset(args)
    print(json.dumps({k: v for k, v in report.items() if k != "skipped"}, indent=2))
    if report["skipped_samples"]:
        print("first_skipped", report["skipped"][:5])


if __name__ == "__main__":
    main()
