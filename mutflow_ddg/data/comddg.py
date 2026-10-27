from __future__ import annotations

import pickle
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .schema import GraphBatch, MutationBatch, MutationSample
from .identity import match_items_to_baseline, parse_record_name, read_allowed_mutation_ids


AA_VOLUME = torch.tensor(
    [88.6, 173.4, 114.1, 111.1, 108.5, 143.8, 138.4, 60.1, 153.2, 166.7,
     166.7, 168.6, 162.9, 189.9, 112.7, 89.0, 116.1, 227.8, 193.6, 140.0],
    dtype=torch.float32,
)
AA_CHARGE = torch.tensor(
    [0, 1, 0, -1, 0, 0, -1, 0, 0.1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    dtype=torch.float32,
)
AA_HYDROPHOBICITY = torch.tensor(
    [1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5,
     3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2],
    dtype=torch.float32,
)
BLOSUM62 = torch.tensor(
    [
        [4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
        [-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
        [-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
        [-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
        [0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
        [-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
        [-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
        [0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
        [-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
        [-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
        [-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
        [-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
        [-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
        [-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
        [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
        [1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
        [0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
        [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
        [-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
        [0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4],
    ],
    dtype=torch.float32,
)


def _tensor(value: Any, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(value, dtype=dtype)


def _infer_residue_type(x: Tensor) -> Tensor | None:
    if x.shape[1] < 20:
        return None
    residue_features = x[:, :20]
    if not torch.all(residue_features.max(dim=-1).values > 0):
        return None
    return residue_features.argmax(dim=-1).long()


def _clean_record_name(value: Any) -> str:
    name = Path(str(value)).name
    return name[:-4] if name.lower().endswith(".pdb") else name


def _pssm_sequence_features(
    record: dict[str, Any],
    feature_dim: int,
) -> Tensor:
    raw_x = _tensor(record["node_features"], torch.float32)
    if raw_x.ndim != 2:
        raise ValueError(
            "PSSM node features must have shape [num_nodes, feature_dim]"
        )
    if raw_x.shape[1] < feature_dim:
        raise ValueError(
            f"PSSM representation requires at least {feature_dim} node "
            f"features, got {raw_x.shape[1]}"
        )
    return raw_x[:, :feature_dim]


def _align_features_to_record(
    features: Tensor,
    record: dict[str, Any],
    record_name: str,
) -> Tensor:
    raw_x = _tensor(record["node_features"], torch.float32)
    if features.shape[0] == raw_x.shape[0]:
        return features
    if raw_x.shape[1] >= 20:
        known_residues = raw_x[:, :20].max(dim=-1).values > 0
        if features.shape[0] == int(known_residues.sum()):
            aligned = features.new_zeros((raw_x.shape[0], features.shape[1]))
            aligned[known_residues] = features
            return aligned
    raise ValueError(
        f"Sequence/graph residue count mismatch for {record_name}: "
        f"{features.shape[0]} vs {raw_x.shape[0]}"
    )


class ESMFeatureStore:
    """Resolve and lazily cache per-residue ESM tensors."""

    def __init__(
        self,
        root: str | Path | Sequence[str | Path],
        max_cache_items: int = 256,
        layout: str = "standard",
    ):
        roots = (
            list(root)
            if isinstance(root, Sequence) and not isinstance(root, (str, Path))
            else [root]
        )
        self.roots = [Path(value) for value in roots]
        self.root = self.roots[0]
        self.max_cache_items = max_cache_items
        if layout not in {"standard", "paired_flat"}:
            raise ValueError(f"Unsupported ESM cache layout: {layout!r}")
        self.layout = layout
        self._cache: OrderedDict[str, Tensor] = OrderedDict()

    def path_for(self, record_name: str) -> Path | None:
        name = _clean_record_name(record_name)
        candidates = [
            candidate
            for root in self.roots
            for candidate in (root / name / "esm2.pt", root / f"{name}_esm2.pt")
        ]
        return next((path for path in candidates if path.is_file()), None)

    def paths_for_item(
        self, item: dict[str, Any]
    ) -> tuple[Path | None, Path | None]:
        if self.layout == "standard":
            return (
                self.path_for(item.get("wild_type_name", "")),
                self.path_for(item.get("mutant_name", "")),
            )

        mutant_name = _clean_record_name(item.get("mutant_name", ""))
        if "_mutant_" not in mutant_name:
            return None, None
        wild_name = mutant_name.replace("_mutant_", "_wild_", 1)
        wild_path = next(
            (
                root / f"{wild_name}_esm2.pt"
                for root in self.roots
                if (root / f"{wild_name}_esm2.pt").is_file()
            ),
            None,
        )
        mutant_path = next(
            (
                root / f"{mutant_name}_esm2.pt"
                for root in self.roots
                if (root / f"{mutant_name}_esm2.pt").is_file()
            ),
            None,
        )
        return (
            wild_path,
            mutant_path,
        )

    def _load_path(self, path: Path) -> Tensor:
        cache_key = str(path)
        if cache_key in self._cache:
            value = self._cache.pop(cache_key)
            self._cache[cache_key] = value
            return value

        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except (TypeError, RuntimeError, pickle.UnpicklingError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                try:
                    value = torch.load(
                        path,
                        map_location="cpu",
                        weights_only=False,
                    )
                except TypeError:
                    value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            value = value.get("features", value.get("representations"))
        if value is None:
            raise ValueError(f"Unsupported ESM cache payload in {path}")
        tensor = _tensor(value, torch.float32)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2:
            raise ValueError(
                f"ESM representation in {path} must have shape [L, D], "
                f"got {tuple(tensor.shape)}"
            )
        self._cache[cache_key] = tensor
        if len(self._cache) > self.max_cache_items:
            self._cache.popitem(last=False)
        return tensor

    def load(self, record_name: str) -> Tensor:
        name = _clean_record_name(record_name)
        path = self.path_for(name)
        if path is None:
            raise FileNotFoundError(
                f"ESM representation not found for {name!r} under {self.root}"
            )
        return self._load_path(path)

    def load_pair(
        self, item: dict[str, Any]
    ) -> tuple[Tensor | None, Tensor | None]:
        wild_path, mutant_path = self.paths_for_item(item)
        return (
            self._load_path(wild_path) if wild_path is not None else None,
            self._load_path(mutant_path) if mutant_path is not None else None,
        )


def graph_from_record(
    record: dict[str, Any],
    sequence_features: Tensor | None = None,
    node_feature_dim: int | None = None,
    ignore_node_features: bool = False,
) -> GraphBatch:
    raw_x = _tensor(record["node_features"], torch.float32)
    x = (
        raw_x.new_zeros((raw_x.shape[0], 1))
        if ignore_node_features
        else raw_x
    )
    if node_feature_dim is not None and not ignore_node_features:
        if x.shape[1] > node_feature_dim:
            raise ValueError(
                f"Cannot reduce {x.shape[1]} node features to {node_feature_dim}"
            )
        if x.shape[1] < node_feature_dim:
            padding = x.new_zeros((x.shape[0], node_feature_dim - x.shape[1]))
            x = torch.cat([x, padding], dim=-1)
    edge_index = _tensor(record["edge_index"], torch.long)
    edge_attr = _tensor(record["edge_features"], torch.float32)
    if sequence_features is None:
        raw_sequence_features = record.get("sequence_features")
        sequence_features = (
            _tensor(raw_sequence_features, torch.float32)
            if raw_sequence_features is not None
            else None
        )
    confidence = record.get("confidence")
    residue_type = record.get("residue_type")
    if residue_type is None:
        inferred_residue_type = _infer_residue_type(x)
    else:
        inferred_residue_type = _tensor(residue_type, torch.long)
    return GraphBatch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros(x.shape[0], dtype=torch.long),
        mutation_index=torch.tensor(
            [int(record.get("mutation_pos", 0))], dtype=torch.long
        ),
        sequence_features=sequence_features,
        residue_type=inferred_residue_type,
        confidence=(
            _tensor(confidence, torch.float32) if confidence is not None else None
        ),
    )


def _align_sequence_features(
    graph: GraphBatch,
    record_name: str,
) -> None:
    features = graph.sequence_features
    if features is None or features.shape[0] == graph.x.shape[0]:
        return

    record = {"node_features": graph.x}
    graph.sequence_features = _align_features_to_record(
        features, record, record_name
    )


def _sequence_record_for_item(
    item: dict[str, Any],
    structural_key: str,
) -> dict[str, Any]:
    sequence_keys = {
        "wild_type": (
            "wild_type_sequence",
            "wild_type_full",
            "wild_type_sequence_full",
        ),
        "mutant": (
            "mutant_sequence",
            "mutant_full",
            "mutant_sequence_full",
        ),
    }
    for key in sequence_keys[structural_key]:
        if key in item:
            return item[key]
    return item[structural_key]


def _align_optional_sequence_features(
    features: Tensor | None,
    record: dict[str, Any],
    record_name: str,
) -> Tensor | None:
    if features is None:
        return None
    return _align_features_to_record(features, record, record_name)


def convert_comddg_item(
    item: dict[str, Any],
    esm_store: ESMFeatureStore | None = None,
    require_esm: bool = False,
    sequence_feature_source: str = "esm",
    sequence_feature_dim: int | None = None,
    node_feature_dim: int | None = None,
    ddg_multiplier: float = 1.0,
    ignore_node_features: bool = False,
    mutation_feature_mode: str = "none",
) -> MutationSample:
    """Translate one existing CoMDDG dictionary without coupling to PyG."""

    wild_name = item.get("wild_type_name", "")
    mutant_name = item.get("mutant_name", "")
    protein_id = wild_name.split("_wild_type")[0]
    wild_sequence_record = _sequence_record_for_item(item, "wild_type")
    mutant_sequence_record = _sequence_record_for_item(item, "mutant")
    has_separate_sequence_records = (
        wild_sequence_record is not item["wild_type"]
        or mutant_sequence_record is not item["mutant"]
    )
    if has_separate_sequence_records and (
        wild_sequence_record is item["wild_type"]
        or mutant_sequence_record is item["mutant"]
    ):
        raise ValueError(
            "Full-sequence records must be supplied for both WT and mutant"
        )
    if sequence_feature_source == "none":
        wild_sequence = mutant_sequence = None
    elif sequence_feature_source in {"esm", "esm_pssm"}:
        wild_sequence, mutant_sequence = (
            esm_store.load_pair(item) if esm_store is not None else (None, None)
        )
        if sequence_feature_source == "esm_pssm":
            if wild_sequence is None or mutant_sequence is None:
                raise ValueError(
                    "ESM representations are required for ESM+PSSM fusion"
                )
            if wild_sequence.shape[1] != mutant_sequence.shape[1]:
                raise ValueError(
                    "WT and mutant ESM feature dimensions must match"
                )
            pssm_dim = sequence_feature_dim - wild_sequence.shape[1]
            if pssm_dim < 1:
                raise ValueError(
                    "ESM+PSSM sequence_feature_dim must exceed ESM dimension"
                )
            wild_sequence = torch.cat(
                [
                    _align_features_to_record(
                        wild_sequence, wild_sequence_record, wild_name
                    ),
                    _pssm_sequence_features(wild_sequence_record, pssm_dim),
                ],
                dim=-1,
            )
            mutant_sequence = torch.cat(
                [
                    _align_features_to_record(
                        mutant_sequence, mutant_sequence_record, mutant_name
                    ),
                    _pssm_sequence_features(mutant_sequence_record, pssm_dim),
                ],
                dim=-1,
            )
        else:
            wild_sequence = _align_optional_sequence_features(
                wild_sequence,
                wild_sequence_record,
                wild_name,
            )
            mutant_sequence = _align_optional_sequence_features(
                mutant_sequence,
                mutant_sequence_record,
                mutant_name,
            )
    elif sequence_feature_source == "pssm":
        if sequence_feature_dim is None:
            raise ValueError(
                "sequence_feature_dim is required for PSSM representations"
            )
        wild_sequence = _pssm_sequence_features(
            wild_sequence_record, sequence_feature_dim
        )
        mutant_sequence = _pssm_sequence_features(
            mutant_sequence_record, sequence_feature_dim
        )
    else:
        raise ValueError(
            f"Unsupported sequence feature source: {sequence_feature_source!r}"
        )
    if require_esm and sequence_feature_source not in {"esm", "esm_pssm"}:
        raise ValueError(
            "require_esm=True requires sequence_feature_source=esm or esm_pssm"
        )
    if require_esm and (wild_sequence is None or mutant_sequence is None):
        raise ValueError(
            "ESM representations are required; configure data.esm_cache_dir"
        )
    wild = graph_from_record(
        item["wild_type"],
        None if has_separate_sequence_records else wild_sequence,
        node_feature_dim=node_feature_dim,
        ignore_node_features=ignore_node_features,
    )
    mutant = graph_from_record(
        item["mutant"],
        None if has_separate_sequence_records else mutant_sequence,
        node_feature_dim=node_feature_dim,
        ignore_node_features=ignore_node_features,
    )
    sequence_wild = sequence_mutant = None
    if has_separate_sequence_records:
        sequence_wild = graph_from_record(
            wild_sequence_record,
            wild_sequence,
            node_feature_dim=node_feature_dim,
            ignore_node_features=ignore_node_features,
        )
        sequence_mutant = graph_from_record(
            mutant_sequence_record,
            mutant_sequence,
            node_feature_dim=node_feature_dim,
            ignore_node_features=ignore_node_features,
        )
    for name, graph in (
        (wild_name, sequence_wild or wild),
        (mutant_name, sequence_mutant or mutant),
    ):
        _align_sequence_features(graph, name)
    mutation_features = (
        _tensor(item["mutation_features"], torch.float32)
        if "mutation_features" in item
        else None
    )
    if mutation_feature_mode == "aa_properties":
        wild_type = wild.residue_type
        mutant_type = mutant.residue_type
        if wild_type is None:
            wild_type = _infer_residue_type(
                _tensor(item["wild_type"]["node_features"], torch.float32)
            )
        if mutant_type is None:
            mutant_type = _infer_residue_type(
                _tensor(item["mutant"]["node_features"], torch.float32)
            )
        if wild_type is None or mutant_type is None:
            raise ValueError("AA-property mutation features require residue types")
        wt = int(wild_type[int(wild.mutation_index[0])])
        mt = int(mutant_type[int(mutant.mutation_index[0])])
        one_hot = torch.zeros(40, dtype=torch.float32)
        one_hot[wt] = 1
        one_hot[20 + mt] = 1
        mutation_features = torch.cat(
            [
                one_hot,
                torch.tensor(
                    [
                        float((AA_VOLUME[mt] - AA_VOLUME[wt]) / 100.0),
                        float(AA_CHARGE[mt] - AA_CHARGE[wt]),
                        float(
                            (AA_HYDROPHOBICITY[mt] - AA_HYDROPHOBICITY[wt])
                            / 5.0
                        ),
                        float(BLOSUM62[wt, mt] / 11.0),
                    ],
                    dtype=torch.float32,
                ),
            ]
        )
    elif mutation_feature_mode != "none":
        raise ValueError(f"unsupported mutation feature mode: {mutation_feature_mode}")
    return MutationSample(
        wild=wild,
        mutant=mutant,
        ddg=torch.tensor(
            float(item["ddg"]) * ddg_multiplier,
            dtype=torch.float32,
        ),
        sequence_wild=sequence_wild,
        sequence_mutant=sequence_mutant,
        mutation_features=mutation_features,
        protein_id=protein_id,
        mutation_name=mutant_name,
        mutation_id=str(item.get("mutation_id", _clean_record_name(mutant_name))),
        num_mutations=int(item.get("num_mutations", 1)),
        epistasis_target=(
            torch.tensor(float(item["epistasis_target"]), dtype=torch.float32)
            * ddg_multiplier
            if "epistasis_target" in item
            else None
        ),
    )


class CoMDDGPairDataset(Dataset):
    """Thin adapter for CoMDDG pickle files.

    Loading is intentionally isolated here. Model and training code only consume
    MutationBatch and therefore remain independent of the legacy repository.
    """

    def __init__(
        self,
        path: str | Path,
        esm_cache_dir: str | Path | Sequence[str | Path] | None = None,
        require_esm: bool = False,
        esm_layout: str = "standard",
        sequence_feature_source: str = "esm",
        sequence_feature_dim: int | None = None,
        node_feature_dim: int | None = None,
        skip_missing_esm: bool = False,
        ddg_multiplier: float = 1.0,
        ignore_node_features: bool = False,
        mutation_feature_mode: str = "none",
        dataset_name: str | None = None,
        manifest_path: str | Path | None = None,
    ):
        with Path(path).open("rb") as handle:
            self.items = pickle.load(handle)
        self.original_size = len(self.items)
        self.skipped_missing_esm = 0
        self.esm_store = (
            ESMFeatureStore(esm_cache_dir, layout=esm_layout)
            if esm_cache_dir is not None
            else None
        )
        self.require_esm = require_esm
        self.sequence_feature_source = sequence_feature_source
        self.sequence_feature_dim = sequence_feature_dim
        self.node_feature_dim = node_feature_dim
        self.ddg_multiplier = ddg_multiplier
        self.ignore_node_features = ignore_node_features
        self.mutation_feature_mode = mutation_feature_mode
        self.dataset_name = dataset_name
        if manifest_path is not None:
            allowed = read_allowed_mutation_ids(manifest_path, dataset_name)
            import csv

            with Path(manifest_path).open(
                "r", newline="", encoding="utf-8-sig"
            ) as handle:
                manifest_rows = [
                    row
                    for row in csv.DictReader(handle)
                    if row.get("mutation_id") in allowed
                ]
            matches = match_items_to_baseline(self.items, manifest_rows)
            retained = []
            for item, row in zip(self.items, matches):
                mutation_id = item.get("mutation_id")
                if mutation_id in allowed:
                    retained.append(item)
                elif row is not None:
                    copied = dict(item)
                    copied["mutation_id"] = row["mutation_id"]
                    copied["source_row_index"] = row.get("source_row_index")
                    retained.append(copied)
            self.items = retained
            if not self.items:
                raise ValueError(
                    f"No samples matched common-subset manifest {manifest_path}"
                )
        if require_esm and self.esm_store is None:
            raise ValueError(
                "require_esm=True but no ESM cache directory was configured"
            )
        if (
            sequence_feature_source == "pssm"
            and sequence_feature_dim is None
        ):
            raise ValueError(
                "sequence_feature_dim is required for PSSM representations"
            )
        if sequence_feature_source not in {
            "none",
            "esm",
            "pssm",
            "esm_pssm",
        }:
            raise ValueError(
                "sequence_feature_source must be one of: "
                "none, esm, pssm, esm_pssm"
            )
        if sequence_feature_source == "pssm" and self.esm_store is not None:
            raise ValueError(
                "esm_cache_dir cannot be used with PSSM representations"
            )
        if self.esm_store is not None:
            missing = self.missing_esm_records()
            if require_esm and missing:
                if skip_missing_esm:
                    retained_items = [
                        item
                        for item in self.items
                        if all(
                            path is not None
                            for path in self.esm_store.paths_for_item(item)
                        )
                    ]
                    self.skipped_missing_esm = (
                        self.original_size - len(retained_items)
                    )
                    self.items = retained_items
                    if not self.items:
                        raise ValueError(
                            "No samples have complete ESM cache entries"
                        )
                else:
                    examples = ", ".join(missing[:5])
                    raise FileNotFoundError(
                        f"{len(missing)} ESM cache entries are missing; "
                        f"examples: {examples}"
                    )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> MutationSample:
        return convert_comddg_item(
            self.items[index],
            esm_store=self.esm_store,
            require_esm=self.require_esm,
            sequence_feature_source=self.sequence_feature_source,
            sequence_feature_dim=self.sequence_feature_dim,
            node_feature_dim=self.node_feature_dim,
            ddg_multiplier=self.ddg_multiplier,
            ignore_node_features=self.ignore_node_features,
            mutation_feature_mode=self.mutation_feature_mode,
        )

    def protein_id_at(self, index: int) -> str:
        wild_name = str(self.items[index].get("wild_type_name", ""))
        return wild_name.split("_wild_type")[0]

    def pair_length_at(self, index: int) -> int:
        item = self.items[index]
        wild_record = _sequence_record_for_item(item, "wild_type")
        mutant_record = _sequence_record_for_item(item, "mutant")
        return max(
            len(wild_record["node_features"]),
            len(mutant_record["node_features"]),
        )

    def missing_esm_records(self) -> list[str]:
        if self.esm_store is None:
            return []
        missing = set()
        for item in self.items:
            paths = self.esm_store.paths_for_item(item)
            names = (
                _clean_record_name(item.get("wild_type_name", "")),
                _clean_record_name(item.get("mutant_name", "")),
            )
            for name, path in zip(names, paths):
                if path is None:
                    missing.add(name)
        return sorted(missing)


def _merge_graphs(graphs: Sequence[GraphBatch]) -> GraphBatch:
    node_offset = 0
    xs, edges, edge_attrs, batches, mutation_indices = [], [], [], [], []
    sequence_features, residue_types, confidences = [], [], []
    has_sequence = all(graph.sequence_features is not None for graph in graphs)
    has_residue = all(graph.residue_type is not None for graph in graphs)
    has_confidence = all(graph.confidence is not None for graph in graphs)

    for graph_id, graph in enumerate(graphs):
        graph.validate()
        xs.append(graph.x)
        edges.append(graph.edge_index + node_offset)
        edge_attrs.append(graph.edge_attr)
        batches.append(torch.full((graph.x.shape[0],), graph_id, dtype=torch.long))
        mutation_indices.append(graph.mutation_index + node_offset)
        if has_sequence:
            sequence_features.append(graph.sequence_features)
        if has_residue:
            residue_types.append(graph.residue_type)
        if has_confidence:
            confidences.append(graph.confidence)
        node_offset += graph.x.shape[0]

    return GraphBatch(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(edges, dim=1),
        edge_attr=torch.cat(edge_attrs, dim=0),
        batch=torch.cat(batches, dim=0),
        mutation_index=torch.cat(mutation_indices, dim=0),
        sequence_features=(
            torch.cat(sequence_features, dim=0) if has_sequence else None
        ),
        residue_type=torch.cat(residue_types, dim=0) if has_residue else None,
        confidence=torch.cat(confidences, dim=0) if has_confidence else None,
    )


def collate_graph_pairs(samples: Sequence[MutationSample]) -> MutationBatch:
    protein_lookup: dict[str, int] = {}
    protein_indices = []
    for sample in samples:
        if sample.protein_id not in protein_lookup:
            protein_lookup[sample.protein_id] = len(protein_lookup)
        protein_indices.append(protein_lookup[sample.protein_id])

    mutation_features = None
    if all(sample.mutation_features is not None for sample in samples):
        mutation_features = torch.stack(
            [sample.mutation_features for sample in samples], dim=0
        )

    epistasis_target = None
    if all(sample.epistasis_target is not None for sample in samples):
        epistasis_target = torch.stack(
            [sample.epistasis_target for sample in samples], dim=0
        )

    has_sequence_pair = [
        sample.sequence_wild is not None or sample.sequence_mutant is not None
        for sample in samples
    ]
    if any(has_sequence_pair) and not all(
        sample.sequence_wild is not None and sample.sequence_mutant is not None
        for sample in samples
    ):
        raise ValueError(
            "Batches cannot mix partial full-sequence graph annotations"
        )
    sequence_wild = sequence_mutant = None
    if all(has_sequence_pair):
        sequence_wild = _merge_graphs([sample.sequence_wild for sample in samples])
        sequence_mutant = _merge_graphs(
            [sample.sequence_mutant for sample in samples]
        )

    return MutationBatch(
        wild=_merge_graphs([sample.wild for sample in samples]),
        mutant=_merge_graphs([sample.mutant for sample in samples]),
        ddg=torch.stack([sample.ddg for sample in samples]),
        sequence_wild=sequence_wild,
        sequence_mutant=sequence_mutant,
        mutation_features=mutation_features,
        protein_index=torch.tensor(protein_indices, dtype=torch.long),
        num_mutations=torch.tensor(
            [sample.num_mutations for sample in samples], dtype=torch.long
        ),
        epistasis_target=epistasis_target,
        mutation_ids=[sample.mutation_id for sample in samples],
    )


def collate_graphs(graphs: Sequence[GraphBatch]) -> GraphBatch:
    return _merge_graphs(graphs)
