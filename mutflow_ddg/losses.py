from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import LossConfig
from .data.schema import MutationBatch


def ddg_class_targets(
    ddg: Tensor, neutral_threshold: float, convention: str
) -> Tensor:
    """Return 0=stabilizing, 1=neutral, 2=destabilizing."""

    if convention == "positive_destabilizing":
        signed = ddg
    elif convention == "negative_destabilizing":
        signed = -ddg
    else:
        raise ValueError(f"Unsupported DDG convention: {convention}")
    targets = torch.ones_like(signed, dtype=torch.long)
    targets[signed < -neutral_threshold] = 0
    targets[signed > neutral_threshold] = 2
    return targets


def pairwise_ranking_loss(
    scores: Tensor,
    ddg: Tensor,
    protein_index: Tensor | None,
    convention: str,
    minimum_gap: float = 0.1,
) -> Tensor:
    if protein_index is None or scores.numel() < 2:
        return scores.new_zeros(())
    utility = -ddg if convention == "positive_destabilizing" else ddg
    losses = []
    for protein_id in protein_index.unique():
        indices = torch.where(protein_index == protein_id)[0]
        if indices.numel() < 2:
            continue
        for left in range(indices.numel()):
            for right in range(left + 1, indices.numel()):
                i, j = indices[left], indices[right]
                utility_gap = utility[i] - utility[j]
                if utility_gap.abs() < minimum_gap:
                    continue
                target = utility_gap.sign()
                losses.append(F.softplus(-target * (scores[i] - scores[j])))
    return torch.stack(losses).mean() if losses else scores.new_zeros(())


class MutFlowObjective(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        output: dict[str, Tensor],
        batch: MutationBatch,
        reverse_output: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        error = output["ddg"] - batch.ddg
        losses: dict[str, Tensor] = {}
        losses["regression"] = F.huber_loss(
            output["ddg"],
            batch.ddg,
            delta=self.config.huber_delta,
        )
        class_target = ddg_class_targets(
            batch.ddg,
            self.config.neutral_threshold,
            self.config.ddg_convention,
        )
        losses["classification"] = F.cross_entropy(
            output["class_logits"], class_target
        )
        losses["ranking"] = pairwise_ranking_loss(
            output["rank_score"],
            batch.ddg,
            batch.protein_index,
            self.config.ddg_convention,
        )
        losses["flow"] = F.mse_loss(
            output["predicted_velocity"], output["target_velocity"]
        )
        losses["transport"] = F.mse_loss(output["z_flow"], output["z1"])
        losses["uncertainty"] = 0.5 * (
            torch.exp(-output["log_variance"]) * error.square()
            + output["log_variance"]
        ).mean()
        losses["symmetry"] = (
            (output["ddg"] + reverse_output["ddg"]).square().mean()
            if reverse_output is not None
            else output["ddg"].new_zeros(())
        )
        if batch.epistasis_target is not None:
            multi_mask = (
                batch.num_mutations > 1
                if batch.num_mutations is not None
                else torch.ones_like(batch.ddg, dtype=torch.bool)
            )
            losses["epistasis"] = (
                F.huber_loss(
                    output["epistasis"][multi_mask],
                    batch.epistasis_target[multi_mask],
                )
                if multi_mask.any()
                else output["ddg"].new_zeros(())
            )
        else:
            losses["epistasis"] = output["ddg"].new_zeros(())

        total = (
            self.config.regression * losses["regression"]
            + self.config.classification * losses["classification"]
            + self.config.ranking * losses["ranking"]
            + self.config.symmetry * losses["symmetry"]
            + self.config.transport * losses["transport"]
            + self.config.uncertainty * losses["uncertainty"]
            + self.config.flow * losses["flow"]
            + self.config.epistasis * losses["epistasis"]
        )
        losses["total"] = total
        return total, losses


def structure_pretraining_loss(
    output: dict[str, Tensor], graph
) -> tuple[Tensor, dict[str, Tensor]]:
    if graph.residue_type is None:
        raise ValueError("Structure pretraining requires graph.residue_type")
    center_target = graph.residue_type[graph.mutation_index]
    central = F.cross_entropy(output["central_logits"], center_target)
    return central, {"total": central, "central_residue": central}


def structure_flow_pretraining_loss(
    output: dict[str, Tensor],
    graph,
    central_weight: float = 1.0,
    flow_weight: float = 1.0,
    transport_weight: float = 0.1,
    contrastive_weight: float = 0.2,
    contrastive_temperature: float = 0.1,
    replacement_weight: float = 0.5,
    geometry_weight: float = 0.2,
    contact_weight: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    if graph.residue_type is None:
        raise ValueError("Structure flow pretraining requires graph.residue_type")
    center_target = graph.residue_type[graph.mutation_index]
    central = F.cross_entropy(output["central_logits"], center_target)
    flow = F.mse_loss(output["predicted_velocity"], output["target_velocity"])
    transport = F.mse_loss(output["z_flow"], output["z1"])
    contrastive = _symmetric_contrastive_loss(
        output["corrupted_graph"],
        output["clean_graph"],
        temperature=contrastive_temperature,
    )
    total = (
        central_weight * central
        + flow_weight * flow
        + transport_weight * transport
        + contrastive_weight * contrastive
    )
    parts = {
        "total": total,
        "central_residue": central,
        "flow": flow,
        "transport": transport,
        "contrastive": contrastive,
    }
    if "replacement_logits" in output:
        replacement = F.cross_entropy(
            output["replacement_logits"], output["replacement_target"]
        )
        total = total + replacement_weight * replacement
        parts["replacement"] = replacement
    if "denoised_distances" in output:
        geometry = F.smooth_l1_loss(
            output["denoised_distances"], output["clean_distances"]
        )
        contact_target = (output["clean_distances"][:, 0] < 8.0).float()
        contact = F.binary_cross_entropy_with_logits(
            output["contact_logits"], contact_target
        )
        total = total + geometry_weight * geometry + contact_weight * contact
        parts["geometry"] = geometry
        parts["contact"] = contact
    parts["total"] = total
    return total, parts


def _symmetric_contrastive_loss(
    left: Tensor,
    right: Tensor,
    temperature: float,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    logits = left @ right.transpose(0, 1) / temperature
    targets = torch.arange(left.shape[0], device=left.device)
    return 0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.transpose(0, 1), targets)
    )
