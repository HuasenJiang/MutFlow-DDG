from __future__ import annotations

import torch
from torch import Tensor, nn


class PredictionHeads(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        input_dim = latent_dim * 5
        self.shared = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
        )
        self.ddg_mean = nn.Linear(latent_dim, 1)
        self.ddg_log_variance = nn.Linear(latent_dim, 1)
        self.stability_class = nn.Linear(latent_dim, num_classes)
        self.ranking = nn.Linear(latent_dim, 1)
        self.epistasis = nn.Linear(latent_dim, 1)

    def forward(
        self,
        z0: Tensor,
        z1: Tensor,
        z_flow: Tensor,
        condition: Tensor,
    ) -> dict[str, Tensor]:
        hidden = self.shared(
            torch.cat([z0, z1, z1 - z0, z_flow - z0, condition], dim=-1)
        )
        return {
            "ddg": self.ddg_mean(hidden).squeeze(-1),
            "log_variance": self.ddg_log_variance(hidden).squeeze(-1).clamp(-8, 8),
            "class_logits": self.stability_class(hidden),
            "rank_score": self.ranking(hidden).squeeze(-1),
            "epistasis": self.epistasis(hidden).squeeze(-1),
            "head_embedding": hidden,
        }
