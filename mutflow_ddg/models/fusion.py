from __future__ import annotations

import torch
from torch import Tensor, nn


class UnimodalStateProjection(nn.Module):
    """Project one encoder branch into the latent flow state space."""

    def __init__(self, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        return self.network(embedding)


class StateCrossModalFusion(nn.Module):
    """Use a mutation-centered query to fuse sequence and structure tokens."""

    def __init__(self, hidden_dim: int, latent_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.sequence_projection = nn.Linear(hidden_dim, latent_dim)
        self.structure_projection = nn.Linear(hidden_dim, latent_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, sequence_embedding: Tensor, structure_embedding: Tensor
    ) -> tuple[Tensor, Tensor]:
        query = self.query(
            torch.cat([sequence_embedding, structure_embedding], dim=-1)
        ).unsqueeze(1)
        tokens = torch.stack(
            [
                self.sequence_projection(sequence_embedding),
                self.structure_projection(structure_embedding),
            ],
            dim=1,
        )
        attended, weights = self.cross_attention(query, tokens, tokens)
        fused = self.output(
            torch.cat([query.squeeze(1), attended.squeeze(1)], dim=-1)
        )
        return fused, weights.squeeze(1)


class MutationConditionEncoder(nn.Module):
    """Build c from WT context, directional branch change and metadata."""

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        mutation_dim: int,
        dropout: float,
    ):
        super().__init__()
        input_dim = latent_dim + hidden_dim * 2 + mutation_dim
        self.mutation_dim = mutation_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(
        self,
        z_wild: Tensor,
        branch_wild: Tensor,
        branch_mutant: Tensor,
        mutation_features: Tensor | None,
    ) -> Tensor:
        features = [
            z_wild,
            branch_mutant - branch_wild,
            branch_mutant * branch_wild,
        ]
        if self.mutation_dim > 0:
            if mutation_features is None:
                mutation_features = z_wild.new_zeros(
                    (z_wild.shape[0], self.mutation_dim)
                )
            features.append(mutation_features)
        return self.network(torch.cat(features, dim=-1))
