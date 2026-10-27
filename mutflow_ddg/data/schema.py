from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor


def reverse_mutation_features(features: Tensor | None) -> Tensor | None:
    if features is None:
        return None
    if features.shape[-1] != 44:
        return -features
    reversed_features = features.clone()
    reversed_features[..., :20] = features[..., 20:40]
    reversed_features[..., 20:40] = features[..., :20]
    reversed_features[..., 40:43] = -features[..., 40:43]
    # BLOSUM62 is symmetric, so the final score is direction-invariant.
    reversed_features[..., 43] = features[..., 43]
    return reversed_features


@dataclass
class GraphBatch:
    """Dependency-free sparse graph batch used by all MutFlow-DDG modules."""

    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    batch: Tensor
    mutation_index: Tensor
    sequence_features: Tensor | None = None
    residue_type: Tensor | None = None
    confidence: Tensor | None = None

    @property
    def num_graphs(self) -> int:
        return int(self.mutation_index.numel())

    def to(self, device: torch.device | str) -> "GraphBatch":
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if isinstance(value, Tensor) else value
        return GraphBatch(**values)

    def validate(self) -> None:
        if self.x.ndim != 2:
            raise ValueError("x must have shape [num_nodes, node_dim]")
        if self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if self.edge_attr.shape[0] != self.edge_index.shape[1]:
            raise ValueError("edge_attr and edge_index disagree on num_edges")
        if self.batch.shape[0] != self.x.shape[0]:
            raise ValueError("batch must contain one graph id per node")
        if self.mutation_index.numel() == 0:
            raise ValueError("each graph batch needs mutation indices")
        if self.mutation_index.max().item() >= self.x.shape[0]:
            raise ValueError("mutation_index is outside the node tensor")
        if self.sequence_features is not None:
            if self.sequence_features.ndim != 2:
                raise ValueError(
                    "sequence_features must have shape [num_nodes, sequence_dim]"
                )
            if self.sequence_features.shape[0] != self.x.shape[0]:
                raise ValueError(
                    "sequence_features and x disagree on num_nodes"
                )
        if self.residue_type is not None:
            if self.residue_type.ndim != 1:
                raise ValueError("residue_type must have shape [num_nodes]")
            if self.residue_type.shape[0] != self.x.shape[0]:
                raise ValueError("residue_type and x disagree on num_nodes")


@dataclass
class MutationSample:
    wild: GraphBatch
    mutant: GraphBatch
    ddg: Tensor
    sequence_wild: GraphBatch | None = None
    sequence_mutant: GraphBatch | None = None
    mutation_features: Tensor | None = None
    protein_id: str = ""
    mutation_name: str = ""
    mutation_id: str = ""
    num_mutations: int = 1
    epistasis_target: Tensor | None = None

    @property
    def wild_for_sequence(self) -> GraphBatch:
        return self.sequence_wild if self.sequence_wild is not None else self.wild

    @property
    def mutant_for_sequence(self) -> GraphBatch:
        return (
            self.sequence_mutant
            if self.sequence_mutant is not None
            else self.mutant
        )


@dataclass
class MutationBatch:
    wild: GraphBatch
    mutant: GraphBatch
    ddg: Tensor
    sequence_wild: GraphBatch | None = None
    sequence_mutant: GraphBatch | None = None
    mutation_features: Tensor | None = None
    protein_index: Tensor | None = None
    num_mutations: Tensor | None = None
    epistasis_target: Tensor | None = None
    mutation_ids: list[str] | None = None

    @property
    def wild_for_sequence(self) -> GraphBatch:
        return self.sequence_wild if self.sequence_wild is not None else self.wild

    @property
    def mutant_for_sequence(self) -> GraphBatch:
        return (
            self.sequence_mutant
            if self.sequence_mutant is not None
            else self.mutant
        )

    def to(self, device: torch.device | str) -> "MutationBatch":
        return MutationBatch(
            wild=self.wild.to(device),
            mutant=self.mutant.to(device),
            ddg=self.ddg.to(device),
            sequence_wild=(
                self.sequence_wild.to(device)
                if self.sequence_wild is not None
                else None
            ),
            sequence_mutant=(
                self.sequence_mutant.to(device)
                if self.sequence_mutant is not None
                else None
            ),
            mutation_features=(
                self.mutation_features.to(device)
                if self.mutation_features is not None
                else None
            ),
            protein_index=(
                self.protein_index.to(device)
                if self.protein_index is not None
                else None
            ),
            num_mutations=(
                self.num_mutations.to(device)
                if self.num_mutations is not None
                else None
            ),
            epistasis_target=(
                self.epistasis_target.to(device)
                if self.epistasis_target is not None
                else None
            ),
            mutation_ids=self.mutation_ids,
        )

    def reversed(self) -> "MutationBatch":
        return MutationBatch(
            wild=self.mutant,
            mutant=self.wild,
            ddg=-self.ddg,
            sequence_wild=self.sequence_mutant,
            sequence_mutant=self.sequence_wild,
            mutation_features=(
                reverse_mutation_features(self.mutation_features)
            ),
            protein_index=self.protein_index,
            num_mutations=self.num_mutations,
            epistasis_target=(
                -self.epistasis_target
                if self.epistasis_target is not None
                else None
            ),
            mutation_ids=self.mutation_ids,
        )
