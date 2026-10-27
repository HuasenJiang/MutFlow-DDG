from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TimeEmbedding(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        half_dim = max(output_dim // 2, 1)
        frequencies = torch.exp(
            torch.linspace(0, math.log(1000.0), half_dim)
        )
        self.register_buffer("frequencies", frequencies)
        self.projection = nn.Linear(half_dim * 2, output_dim)

    def forward(self, time: Tensor) -> Tensor:
        angles = time * self.frequencies.unsqueeze(0)
        return self.projection(torch.cat([angles.sin(), angles.cos()], dim=-1))


class ConditionalVectorField(nn.Module):
    def __init__(self, latent_dim: int, dropout: float):
        super().__init__()
        self.time_embedding = TimeEmbedding(latent_dim)
        self.network = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim * 3),
            nn.LayerNorm(latent_dim * 3),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, state: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        return self.network(
            torch.cat([state, self.time_embedding(time), condition], dim=-1)
        )


class LatentConditionalFlow(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_steps: int,
        noise_scale: float,
        dropout: float,
    ):
        super().__init__()
        self.vector_field = ConditionalVectorField(latent_dim, dropout)
        self.num_steps = num_steps
        self.noise_scale = noise_scale

    def sample_training_path(
        self, z0: Tensor, z1: Tensor, condition: Tensor
    ) -> dict[str, Tensor]:
        time = torch.rand(z0.shape[0], 1, device=z0.device, dtype=z0.dtype)
        noise = torch.randn_like(z0)
        sigma = self.noise_scale * time * (1.0 - time)
        state = (1.0 - time) * z0 + time * z1 + sigma * noise
        target_velocity = z1 - z0
        predicted_velocity = self.vector_field(state, time, condition)
        return {
            "time": time,
            "state": state,
            "target_velocity": target_velocity,
            "predicted_velocity": predicted_velocity,
        }

    def integrate(self, z0: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        state = z0
        trajectory = [state]
        step_size = 1.0 / self.num_steps
        for step in range(self.num_steps):
            time = state.new_full((state.shape[0], 1), step * step_size)
            velocity = self.vector_field(state, time, condition)
            state = state + step_size * velocity
            trajectory.append(state)
        return state, torch.stack(trajectory, dim=1)

