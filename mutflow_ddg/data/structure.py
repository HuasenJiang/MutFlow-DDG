from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .comddg import graph_from_record
from .schema import GraphBatch


class StructureGraphDataset(Dataset):
    """Load unlabeled graphs and sample one Pythia-style local center crop."""

    def __init__(
        self,
        path: str | Path,
        deduplicate_proteins: bool = True,
        local_crop_radius: float | None = None,
        local_crop_max_nodes: int | None = None,
        local_crop_neighbors: int | None = None,
        centers_per_protein: int = 1,
        fixed_centers: bool = False,
        center_seed: int = 0,
    ):
        path = Path(path)
        if local_crop_radius is not None and local_crop_radius <= 0:
            raise ValueError("local_crop_radius must be positive or null")
        if local_crop_max_nodes is not None and local_crop_max_nodes < 1:
            raise ValueError("local_crop_max_nodes must be positive or null")
        if local_crop_neighbors is not None and local_crop_neighbors < 1:
            raise ValueError("local_crop_neighbors must be positive or null")
        if centers_per_protein < 1:
            raise ValueError("centers_per_protein must be positive")
        self.records: list[dict[str, Any]] | None = None
        self.shard_paths: list[Path] = []
        self.shard_indices: list[tuple[int, int]] = []
        self._cached_shard_id: int | None = None
        self._cached_shard_records: list[dict[str, Any]] | None = None
        self.local_crop_radius = local_crop_radius
        self.local_crop_max_nodes = local_crop_max_nodes
        self.local_crop_neighbors = local_crop_neighbors
        self.centers_per_protein = centers_per_protein
        self.fixed_centers = fixed_centers
        self.center_seed = center_seed

        if path.is_dir() or path.suffix == ".txt":
            self._load_sharded_index(path, deduplicate_proteins)
            if not self.shard_indices:
                raise ValueError(f"No structure graph records found in {path}")
            self.shard_indices = [
                entry
                for entry in self.shard_indices
                for _ in range(self.centers_per_protein)
            ]
            return

        with path.open("rb") as handle:
            items = pickle.load(handle)
        records = []
        seen: set[str] = set()
        for index, item in enumerate(items):
            record, name = _record_and_name(item, index)
            protein_id = name.split("_wild_type")[0]
            if deduplicate_proteins and protein_id in seen:
                continue
            seen.add(protein_id)
            records.append(record)
        if not records:
            raise ValueError(f"No structure graph records found in {path}")
        self.records = records

    def _load_sharded_index(
        self,
        path: Path,
        deduplicate_proteins: bool,
    ) -> None:
        self.shard_paths = _resolve_shards(path)
        seen: set[str] = set()
        for shard_id, shard_path in enumerate(self.shard_paths):
            with shard_path.open("rb") as handle:
                items = pickle.load(handle)
            for local_index, item in enumerate(items):
                _, name = _record_and_name(item, local_index)
                protein_id = name.split("_wild_type")[0]
                if deduplicate_proteins and protein_id in seen:
                    continue
                seen.add(protein_id)
                self.shard_indices.append((shard_id, local_index))

    def _load_shard(self, shard_id: int) -> list[dict[str, Any]]:
        if (
            self._cached_shard_id == shard_id
            and self._cached_shard_records is not None
        ):
            return self._cached_shard_records
        with self.shard_paths[shard_id].open("rb") as handle:
            items = pickle.load(handle)
        records = [
            _record_and_name(item, local_index)[0]
            for local_index, item in enumerate(items)
        ]
        self._cached_shard_id = shard_id
        self._cached_shard_records = records
        return records

    def __len__(self) -> int:
        if self.records is not None:
            return len(self.records) * self.centers_per_protein
        return len(self.shard_indices)

    def __getitem__(self, index: int) -> GraphBatch:
        if self.records is None:
            shard_id, local_index = self.shard_indices[index]
            record = self._load_shard(shard_id)[local_index]
        else:
            record = self.records[index // self.centers_per_protein]
        graph = graph_from_record(record)
        generator = None
        if self.fixed_centers:
            generator = torch.Generator().manual_seed(self.center_seed + index)
        center = torch.randint(
            graph.x.shape[0],
            (1,),
            dtype=torch.long,
            generator=generator,
        )
        if self.local_crop_neighbors is not None:
            ca_coords = record.get("ca_coords")
            if ca_coords is None:
                name = record.get("name", index)
                raise ValueError(
                    "Local structure pretraining requires ca_coords in graph "
                    f"records; re-run convert_cath_pdb_to_graphs.py for {name}"
                )
            graph = _crop_graph_by_ca_neighbors(
                graph,
                ca_coords,
                int(center.item()),
                self.local_crop_neighbors,
            )
        elif self.local_crop_radius is not None:
            ca_coords = record.get("ca_coords")
            if ca_coords is None:
                name = record.get("name", index)
                raise ValueError(
                    "Local structure pretraining requires ca_coords in graph "
                    f"records; re-run convert_cath_pdb_to_graphs.py for {name}"
                )
            graph = _crop_graph_by_ca_radius(
                graph,
                ca_coords,
                int(center.item()),
                self.local_crop_radius,
                self.local_crop_max_nodes,
            )
        else:
            graph.mutation_index = center
        graph.sequence_features = None
        if graph.residue_type is None:
            raise ValueError(
                "Structure pretraining needs residue identities in the first "
                "20 node-feature columns or an explicit residue_type field"
            )
        return graph


def _record_and_name(
    item: dict[str, Any],
    fallback_index: int,
) -> tuple[dict[str, Any], str]:
    if "wild_type" in item:
        return item["wild_type"], str(item.get("wild_type_name", fallback_index))
    return item, str(item.get("name", fallback_index))


def _resolve_shards(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.pkl"))
    shards = [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    invalid = [shard for shard in shards if shard.suffix != ".pkl"]
    if invalid:
        raise ValueError(
            "StructureGraphDataset shard manifests must list graph .pkl "
            "files. Convert raw PDB/mmCIF files before pretraining; first "
            f"invalid entry: {invalid[0]}"
        )
    return shards


def _crop_graph_by_ca_radius(
    graph: GraphBatch,
    ca_coords: Any,
    center: int,
    radius: float,
    max_nodes: int | None,
) -> GraphBatch:
    ca = torch.as_tensor(ca_coords, dtype=torch.float32)
    if ca.ndim != 2 or ca.shape[1] != 3:
        raise ValueError("ca_coords must have shape [num_nodes, 3]")
    if ca.shape[0] != graph.x.shape[0]:
        raise ValueError(
            "ca_coords and node_features disagree on num_nodes: "
            f"{ca.shape[0]} vs {graph.x.shape[0]}"
        )
    if center < 0 or center >= ca.shape[0]:
        raise ValueError("center index is outside the graph")

    distances = torch.linalg.norm(ca - ca[center], dim=-1)
    selected = torch.nonzero(distances <= radius, as_tuple=False).flatten()
    if max_nodes is not None and selected.numel() > max_nodes:
        nearest = torch.argsort(distances[selected])[:max_nodes]
        selected = selected[nearest]
    selected = torch.sort(selected.unique()).values
    if not (selected == center).any():
        selected = torch.sort(torch.cat([selected, torch.tensor([center])])).values

    remap = torch.full((graph.x.shape[0],), -1, dtype=torch.long)
    remap[selected] = torch.arange(selected.numel(), dtype=torch.long)
    edge_mask = (
        (remap[graph.edge_index[0]] >= 0)
        & (remap[graph.edge_index[1]] >= 0)
    )
    edge_index = remap[graph.edge_index[:, edge_mask]]
    edge_attr = graph.edge_attr[edge_mask]
    mutation_index = remap[torch.tensor([center], dtype=torch.long)]

    return GraphBatch(
        x=graph.x[selected],
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros(selected.numel(), dtype=torch.long),
        mutation_index=mutation_index,
        sequence_features=(
            graph.sequence_features[selected]
            if graph.sequence_features is not None
            else None
        ),
        residue_type=(
            graph.residue_type[selected] if graph.residue_type is not None else None
        ),
        confidence=(
            graph.confidence[selected] if graph.confidence is not None else None
        ),
    )


def _crop_graph_by_ca_neighbors(
    graph: GraphBatch,
    ca_coords: Any,
    center: int,
    neighbors: int,
) -> GraphBatch:
    ca = torch.as_tensor(ca_coords, dtype=torch.float32)
    if ca.ndim != 2 or ca.shape[1] != 3:
        raise ValueError("ca_coords must have shape [num_nodes, 3]")
    if ca.shape[0] != graph.x.shape[0]:
        raise ValueError(
            "ca_coords and node_features disagree on num_nodes: "
            f"{ca.shape[0]} vs {graph.x.shape[0]}"
        )
    if center < 0 or center >= ca.shape[0]:
        raise ValueError("center index is outside the graph")

    distances = torch.linalg.norm(ca - ca[center], dim=-1)
    order = torch.argsort(distances)
    selected = order[: min(neighbors + 1, order.numel())]
    selected = torch.sort(selected.unique()).values
    if not (selected == center).any():
        selected = torch.sort(torch.cat([selected, torch.tensor([center])])).values

    remap = torch.full((graph.x.shape[0],), -1, dtype=torch.long)
    remap[selected] = torch.arange(selected.numel(), dtype=torch.long)
    edge_mask = (
        (remap[graph.edge_index[0]] >= 0)
        & (remap[graph.edge_index[1]] >= 0)
    )
    edge_index = remap[graph.edge_index[:, edge_mask]]
    edge_attr = graph.edge_attr[edge_mask]
    mutation_index = remap[torch.tensor([center], dtype=torch.long)]

    return GraphBatch(
        x=graph.x[selected],
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros(selected.numel(), dtype=torch.long),
        mutation_index=mutation_index,
        sequence_features=(
            graph.sequence_features[selected]
            if graph.sequence_features is not None
            else None
        ),
        residue_type=(
            graph.residue_type[selected] if graph.residue_type is not None else None
        ),
        confidence=(
            graph.confidence[selected] if graph.confidence is not None else None
        ),
    )
