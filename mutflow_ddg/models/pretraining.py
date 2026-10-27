from __future__ import annotations

import torch
from torch import Tensor, nn

from mutflow_ddg.data.schema import GraphBatch

from .encoders import StructureEncoder
from .flow import LatentConditionalFlow


class StructurePretrainingModel(nn.Module):
    """Predict a center residue from a corrupted local structural environment."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_residue_types: int = 20,
        num_input_tokens: int = 22,
        num_random_tokens: int = 21,
        mask_token_index: int = 21,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = StructureEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.central_residue_head = nn.Linear(hidden_dim, num_residue_types)
        self.num_residue_types = num_residue_types
        self.num_input_tokens = num_input_tokens
        self.num_random_tokens = num_random_tokens
        self.mask_token_index = mask_token_index
        if not 0 <= mask_token_index < num_input_tokens:
            raise ValueError("mask_token_index must be inside num_input_tokens")
        if num_residue_types > num_input_tokens:
            raise ValueError("num_residue_types cannot exceed num_input_tokens")
        if not 1 <= num_random_tokens <= num_input_tokens:
            raise ValueError("num_random_tokens must be inside num_input_tokens")

    def corrupt_graph(
        self,
        graph: GraphBatch,
        mask_ratio: float,
        random_replace_ratio: float,
        edge_noise_std: float,
    ) -> GraphBatch:
        if mask_ratio < 0 or random_replace_ratio < 0:
            raise ValueError("Corruption ratios must be non-negative")
        if mask_ratio + random_replace_ratio > 1:
            raise ValueError("Corruption ratios must sum to at most one")
        if graph.x.shape[-1] < self.num_input_tokens:
            raise ValueError(
                "Node features do not contain the expected Pythia token prefix"
            )
        x = graph.x.clone()
        centers = graph.mutation_index
        draw = torch.rand(centers.numel(), device=x.device)
        masked = draw < mask_ratio
        replaced = (draw >= mask_ratio) & (
            draw < mask_ratio + random_replace_ratio
        )
        x[centers[masked], : self.num_input_tokens] = 0
        x[centers[masked], self.mask_token_index] = 1
        if replaced.any():
            replaced_centers = centers[replaced]
            random_tokens = torch.randint(
                self.num_random_tokens,
                (replaced_centers.numel(),),
                device=x.device,
            )
            x[replaced_centers, : self.num_input_tokens] = 0
            x[replaced_centers, random_tokens] = 1
        edge_attr = graph.edge_attr
        if edge_noise_std > 0 and edge_attr.numel() > 0:
            edge_attr = edge_attr.clone()
            distance_dims = min(25, edge_attr.shape[-1])
            edge_attr[:, :distance_dims] += (
                torch.randn_like(edge_attr[:, :distance_dims]) * edge_noise_std
            )
        return GraphBatch(
            x=x,
            edge_index=graph.edge_index,
            edge_attr=edge_attr,
            batch=graph.batch,
            mutation_index=graph.mutation_index,
            sequence_features=None,
            residue_type=graph.residue_type,
            confidence=graph.confidence,
        )

    def forward(
        self,
        graph: GraphBatch,
        mask_ratio: float = 0.85,
        random_replace_ratio: float = 0.15,
        edge_noise_std: float = 0.5,
        pseudo_mutation: bool = False,
    ) -> dict[str, Tensor]:
        corrupted = self.corrupt_graph(
            graph,
            mask_ratio=mask_ratio,
            random_replace_ratio=random_replace_ratio,
            edge_noise_std=edge_noise_std,
        )
        encoded = self.encoder(corrupted)
        return {
            "central_logits": self.central_residue_head(encoded["graph"]),
            **encoded,
        }


class StructureFlowPretrainingModel(nn.Module):
    """Learn a flow from corrupted to clean local structural representations."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        latent_dim: int | None = None,
        flow_steps: int = 8,
        flow_noise_scale: float = 0.05,
        num_residue_types: int = 20,
        num_input_tokens: int = 22,
        num_random_tokens: int = 21,
        mask_token_index: int = 21,
        dropout: float = 0.1,
    ):
        super().__init__()
        latent_dim = latent_dim or hidden_dim
        self.encoder = StructureEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.state_projection = (
            nn.Identity()
            if latent_dim == hidden_dim
            else nn.Sequential(
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
            )
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.flow = LatentConditionalFlow(
            latent_dim=latent_dim,
            num_steps=flow_steps,
            noise_scale=flow_noise_scale,
            dropout=dropout,
        )
        self.central_residue_head = nn.Linear(hidden_dim, num_residue_types)
        self.replacement_head = nn.Linear(hidden_dim * 2, num_residue_types)
        self.geometry_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 25),
        )
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.num_residue_types = num_residue_types
        self.num_input_tokens = num_input_tokens
        self.num_random_tokens = num_random_tokens
        self.mask_token_index = mask_token_index
        if not 0 <= mask_token_index < num_input_tokens:
            raise ValueError("mask_token_index must be inside num_input_tokens")
        if num_residue_types > num_input_tokens:
            raise ValueError("num_residue_types cannot exceed num_input_tokens")
        if not 1 <= num_random_tokens <= num_input_tokens:
            raise ValueError("num_random_tokens must be inside num_input_tokens")

    def corrupt_graph(
        self,
        graph: GraphBatch,
        mask_ratio: float,
        random_replace_ratio: float,
        edge_noise_std: float,
    ) -> GraphBatch:
        return StructurePretrainingModel.corrupt_graph(
            self,
            graph,
            mask_ratio=mask_ratio,
            random_replace_ratio=random_replace_ratio,
            edge_noise_std=edge_noise_std,
        )

    def forward(
        self,
        graph: GraphBatch,
        mask_ratio: float = 0.85,
        random_replace_ratio: float = 0.15,
        edge_noise_std: float = 0.5,
        pseudo_mutation: bool = False,
    ) -> dict[str, Tensor]:
        clean_encoded = self.encoder(graph)
        pseudo_target = None
        if pseudo_mutation:
            corrupted, pseudo_target = self._pseudo_mutant_graph(
                graph, edge_noise_std=edge_noise_std
            )
            corrupted_encoded = self.encoder(corrupted)
            # Downstream semantics are WT -> Mut, so clean WT is the source.
            source_encoded, target_encoded = clean_encoded, corrupted_encoded
        else:
            corrupted = self.corrupt_graph(
                graph,
                mask_ratio=mask_ratio,
                random_replace_ratio=random_replace_ratio,
                edge_noise_std=edge_noise_std,
            )
            corrupted_encoded = self.encoder(corrupted)
            source_encoded, target_encoded = corrupted_encoded, clean_encoded
        z0 = self.state_projection(source_encoded["graph"])
        z1 = self.state_projection(target_encoded["graph"])
        condition = self.condition_projection(z0)
        path = self.flow.sample_training_path(z0, z1, condition)
        z_flow, trajectory = self.flow.integrate(z0, condition)
        edge_source, edge_target = corrupted.edge_index
        edge_pair = torch.cat(
            [
                corrupted_encoded["nodes"][edge_source],
                corrupted_encoded["nodes"][edge_target],
            ],
            dim=-1,
        )
        output = {
            "central_logits": self.central_residue_head(
                corrupted_encoded["graph"]
            ),
            "clean_graph": clean_encoded["graph"],
            "corrupted_graph": corrupted_encoded["graph"],
            "z0": z0,
            "z1": z1,
            "condition": condition,
            "z_flow": z_flow,
            "trajectory": trajectory,
            **path,
        }
        if edge_pair.numel() > 0:
            distance_dims = min(25, graph.edge_attr.shape[-1])
            output["denoised_distances"] = self.geometry_head(edge_pair)[
                :, :distance_dims
            ]
            output["clean_distances"] = graph.edge_attr[:, :distance_dims]
            output["contact_logits"] = self.contact_head(edge_pair).squeeze(-1)
        if pseudo_target is not None:
            output["replacement_logits"] = self.replacement_head(
                torch.cat(
                    [clean_encoded["graph"], corrupted_encoded["graph"]],
                    dim=-1,
                )
            )
            output["replacement_target"] = pseudo_target
        return output

    def _pseudo_mutant_graph(
        self,
        graph: GraphBatch,
        edge_noise_std: float,
    ) -> tuple[GraphBatch, Tensor]:
        if graph.residue_type is None:
            raise ValueError("pseudo mutations require residue types")
        centers = graph.mutation_index
        wild = graph.residue_type[centers]
        draw = torch.randint(
            self.num_residue_types - 1,
            (centers.numel(),),
            device=graph.x.device,
        )
        mutant = draw + (draw >= wild).long()
        x = graph.x.clone()
        x[centers, : self.num_input_tokens] = 0
        x[centers, mutant] = 1
        edge_attr = graph.edge_attr
        if edge_noise_std > 0 and edge_attr.numel() > 0:
            edge_attr = edge_attr.clone()
            distance_dims = min(25, edge_attr.shape[-1])
            edge_attr[:, :distance_dims] += (
                torch.randn_like(edge_attr[:, :distance_dims]) * edge_noise_std
            )
        pseudo = GraphBatch(
            x=x,
            edge_index=graph.edge_index,
            edge_attr=edge_attr,
            batch=graph.batch,
            mutation_index=centers,
            residue_type=graph.residue_type,
            confidence=graph.confidence,
        )
        return pseudo, mutant
