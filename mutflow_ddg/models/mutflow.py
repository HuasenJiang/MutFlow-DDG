from __future__ import annotations

import torch
from torch import Tensor, nn

from mutflow_ddg.config import ModelConfig
from mutflow_ddg.data.schema import MutationBatch

from .encoders import SequenceEncoder, StructureEncoder
from .flow import LatentConditionalFlow
from .fusion import (
    MutationConditionEncoder,
    StateCrossModalFusion,
    UnimodalStateProjection,
)
from .heads import PredictionHeads


class MutFlowDDG(nn.Module):
    """End-to-end latent perturbation flow model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.branch_mode = config.branch_mode
        self.structure_encoder_frozen = False
        self.sequence_encoder = (
            SequenceEncoder(
                sequence_dim=config.sequence_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_sequence_layers,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
                fusion_mode=config.sequence_fusion,
                esm_dim=config.sequence_esm_dim,
                pssm_dim=config.sequence_pssm_dim,
            )
            if self.uses_sequence
            else None
        )
        self.structure_encoder = (
            StructureEncoder(
                node_dim=config.node_dim,
                edge_dim=config.edge_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_structure_layers,
                dropout=config.dropout,
            )
            if self.uses_structure
            else None
        )
        self.state_fusion = (
            StateCrossModalFusion(
                hidden_dim=config.hidden_dim,
                latent_dim=config.latent_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
            )
            if self.branch_mode == "both"
            else None
        )
        self.state_projection = (
            UnimodalStateProjection(
                hidden_dim=config.hidden_dim,
                latent_dim=config.latent_dim,
                dropout=config.dropout,
            )
            if self.branch_mode != "both"
            else None
        )
        self.condition_encoder = MutationConditionEncoder(
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            mutation_dim=config.mutation_dim,
            dropout=config.dropout,
        )
        self.flow = LatentConditionalFlow(
            latent_dim=config.latent_dim,
            num_steps=config.flow_steps,
            noise_scale=config.flow_noise_scale,
            dropout=config.dropout,
        )
        self.heads = PredictionHeads(
            latent_dim=config.latent_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    @property
    def uses_sequence(self) -> bool:
        return self.branch_mode in {"sequence", "both"}

    @property
    def uses_structure(self) -> bool:
        return self.branch_mode in {"structure", "both"}

    def freeze_structure_encoder(self) -> None:
        if self.structure_encoder is None:
            raise ValueError("Cannot freeze a missing structure encoder")
        self.structure_encoder.requires_grad_(False)
        self.structure_encoder.eval()
        self.structure_encoder_frozen = True

    def set_structure_trainable_layers(self, count: int) -> None:
        """Unfreeze the last ``count`` message-passing layers progressively."""

        if self.structure_encoder is None:
            return
        total = len(self.structure_encoder.layers)
        count = max(0, min(int(count), total))
        self.structure_encoder.requires_grad_(False)
        if count > 0:
            for layer in self.structure_encoder.layers[total - count :]:
                layer.requires_grad_(True)
            self.structure_encoder.pool_key.requires_grad_(True)
            self.structure_encoder.pool_query.requires_grad_(True)
            self.structure_encoder.output.requires_grad_(True)
        if count == total:
            self.structure_encoder.input_projection.requires_grad_(True)
        self.structure_encoder_frozen = count == 0

    def train(self, mode: bool = True):
        super().train(mode)
        if self.structure_encoder_frozen and self.structure_encoder is not None:
            self.structure_encoder.eval()
        return self

    def encode_pair(self, batch: MutationBatch) -> dict[str, Tensor]:
        encoded: dict[str, Tensor] = {}
        sequence_wild = sequence_mutant = None
        structure_wild = structure_mutant = None

        if self.sequence_encoder is not None:
            sequence_wild = self.sequence_encoder(batch.wild_for_sequence)
            sequence_mutant = self.sequence_encoder(batch.mutant_for_sequence)
            encoded["sequence_wild"] = sequence_wild["graph"]
            encoded["sequence_mutant"] = sequence_mutant["graph"]
        if self.structure_encoder is not None:
            if self.structure_encoder_frozen:
                with torch.no_grad():
                    structure_wild = self.structure_encoder(batch.wild)
                    structure_mutant = self.structure_encoder(batch.mutant)
            else:
                structure_wild = self.structure_encoder(batch.wild)
                structure_mutant = self.structure_encoder(batch.mutant)
            encoded["structure_wild"] = structure_wild["graph"]
            encoded["structure_mutant"] = structure_mutant["graph"]

        if self.branch_mode == "both":
            if (
                sequence_wild is None
                or sequence_mutant is None
                or structure_wild is None
                or structure_mutant is None
                or self.state_fusion is None
            ):
                raise RuntimeError("Both branch mode was not initialized correctly")
            z0, attention_wild = self.state_fusion(
                sequence_wild["graph"], structure_wild["graph"]
            )
            z1, attention_mutant = self.state_fusion(
                sequence_mutant["graph"], structure_mutant["graph"]
            )
            condition_wild = sequence_wild["graph"]
            condition_mutant = sequence_mutant["graph"]
            encoded["attention_wild"] = attention_wild
            encoded["attention_mutant"] = attention_mutant
        else:
            if self.state_projection is None:
                raise RuntimeError("Single branch mode was not initialized correctly")
            branch_wild = (
                sequence_wild if sequence_wild is not None else structure_wild
            )
            branch_mutant = (
                sequence_mutant
                if sequence_mutant is not None
                else structure_mutant
            )
            if branch_wild is None or branch_mutant is None:
                raise RuntimeError("The selected encoder branch is unavailable")
            condition_wild = branch_wild["graph"]
            condition_mutant = branch_mutant["graph"]
            z0 = self.state_projection(condition_wild)
            z1 = self.state_projection(condition_mutant)

        condition = self.condition_encoder(
            z_wild=z0,
            branch_wild=condition_wild,
            branch_mutant=condition_mutant,
            mutation_features=batch.mutation_features,
        )
        encoded.update({"z0": z0, "z1": z1, "condition": condition})
        return encoded

    def _forward_direction(self, batch: MutationBatch) -> dict[str, Tensor]:
        encoded = self.encode_pair(batch)
        path = self.flow.sample_training_path(
            encoded["z0"], encoded["z1"], encoded["condition"]
        )
        z_flow, trajectory = self.flow.integrate(
            encoded["z0"], encoded["condition"]
        )
        output = self.heads(
            encoded["z0"], encoded["z1"], z_flow, encoded["condition"]
        )
        output.update(encoded)
        output.update(path)
        output["z_flow"] = z_flow
        output["trajectory"] = trajectory
        return output

    def forward(self, batch: MutationBatch) -> dict[str, Tensor]:
        output = self._forward_direction(batch)
        if not self.config.antisymmetric_output:
            return output
        reverse_output = self._forward_direction(batch.reversed())
        output["raw_ddg_forward"] = output["ddg"]
        output["raw_ddg_reverse"] = reverse_output["ddg"]
        output["ddg"] = 0.5 * (output["ddg"] - reverse_output["ddg"])
        output["rank_score"] = 0.5 * (
            output["rank_score"] - reverse_output["rank_score"]
        )
        output["epistasis"] = 0.5 * (
            output["epistasis"] - reverse_output["epistasis"]
        )
        return output
