from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional CLI dependency
    tqdm = None

from mutflow_ddg.config import load_config
from mutflow_ddg.data import StructureGraphDataset, collate_graphs
from mutflow_ddg.device import describe_device, resolve_device
from mutflow_ddg.losses import (
    structure_flow_pretraining_loss,
    structure_pretraining_loss,
)
from mutflow_ddg.models import (
    StructureFlowPretrainingModel,
    StructurePretrainingModel,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss_for_output(output, graph, pretraining):
    if pretraining.objective in {"flow", "mutation_flow"}:
        return structure_flow_pretraining_loss(
            output,
            graph,
            central_weight=pretraining.central_weight,
            flow_weight=pretraining.flow_weight,
            transport_weight=pretraining.transport_weight,
            contrastive_weight=pretraining.contrastive_weight,
            contrastive_temperature=pretraining.contrastive_temperature,
            replacement_weight=pretraining.replacement_weight,
            geometry_weight=pretraining.geometry_weight,
            contact_weight=pretraining.contact_weight,
        )
    return structure_pretraining_loss(output, graph)


class AdaptiveTaskWeighting(nn.Module):
    """Learn bounded positive task weights without changing the loss scale."""

    def __init__(self, task_priors: dict[str, float]):
        super().__init__()
        self.task_names = [
            name for name, weight in task_priors.items() if weight > 0
        ]
        self.register_buffer(
            "priors",
            torch.tensor(
                [task_priors[name] for name in self.task_names],
                dtype=torch.float32,
            ),
        )
        self.log_variances = nn.Parameter(torch.zeros(len(self.task_names)))

    def forward(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = torch.stack([parts[name] for name in self.task_names])
        weights = self._normalized_weights()
        return (weights * losses).sum()

    def _normalized_weights(self) -> torch.Tensor:
        learned = torch.softmax(-self.log_variances, dim=0)
        weights = learned * self.priors
        # Preserve the configured total task weight. This prevents easy losses
        # approaching zero from receiving unbounded weights.
        return weights * (self.priors.sum() / weights.sum().clamp_min(1e-8))

    def weights(self) -> dict[str, float]:
        values = self._normalized_weights().detach().cpu()
        return {
            name: float(value)
            for name, value in zip(self.task_names, values)
        }


def _progress(loader, desc: str, enabled: bool):
    if tqdm is None or not enabled:
        return loader
    return tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)


def _average_losses(loss_sums: dict[str, float], count: int) -> dict[str, float]:
    divisor = max(count, 1)
    return {name: value / divisor for name, value in loss_sums.items()}


def _format_losses(prefix: str, losses: dict[str, float]) -> str:
    order = ["total", "central_residue", "flow", "transport"]
    parts = [
        f"{prefix}_{name}={losses[name]:.4f}"
        for name in order
        if name in losses
    ]
    parts.extend(
        f"{prefix}_{name}={value:.4f}"
        for name, value in sorted(losses.items())
        if name not in order
    )
    return " ".join(parts)


def _accumulate_losses(
    loss_sums: dict[str, float],
    parts: dict[str, torch.Tensor],
    sample_count: int,
) -> None:
    for name, value in parts.items():
        loss_sums[name] = loss_sums.get(name, 0.0) + float(
            value.detach().cpu().item()
        ) * sample_count


class PretrainingMetricAccumulator:
    def __init__(self):
        self.samples = 0
        self.top1 = 0
        self.top5 = 0
        self.embedding_sum = None
        self.embedding_gram = None

    def update(self, output, graph) -> None:
        target = graph.residue_type[graph.mutation_index]
        logits = output["central_logits"].detach()
        self.samples += int(target.numel())
        self.top1 += int((logits.argmax(dim=-1) == target).sum())
        self.top5 += int(
            (logits.topk(min(5, logits.shape[-1]), dim=-1).indices == target[:, None])
            .any(dim=-1)
            .sum()
        )
        embedding = output.get("corrupted_graph", output.get("graph")).detach()
        embedding = embedding.float().cpu()
        vector_sum = embedding.sum(dim=0)
        gram = embedding.transpose(0, 1) @ embedding
        self.embedding_sum = (
            vector_sum if self.embedding_sum is None else self.embedding_sum + vector_sum
        )
        self.embedding_gram = (
            gram if self.embedding_gram is None else self.embedding_gram + gram
        )

    def compute(self, central_loss: float) -> dict[str, float]:
        count = max(self.samples, 1)
        result = {
            "top1_accuracy": self.top1 / count,
            "top5_accuracy": self.top5 / count,
            "perplexity": math.exp(min(central_loss, 20.0)),
        }
        if self.samples > 1:
            covariance = (
                self.embedding_gram
                - torch.outer(self.embedding_sum, self.embedding_sum) / self.samples
            ) / (self.samples - 1)
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
            probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            result["embedding_effective_rank"] = float(entropy.exp())
        else:
            result["embedding_effective_rank"] = 0.0
        return result


class ShardedBatchSampler:
    """Shuffle by shard, then within shard, to avoid repeated giant pickle loads."""

    def __init__(
        self,
        groups: dict[int, list[int]],
        batch_size: int,
        seed: int,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.groups = {key: list(indices) for key, indices in groups.items()}
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        shard_ids = list(self.groups)
        rng.shuffle(shard_ids)
        for shard_id in shard_ids:
            indices = list(self.groups[shard_id])
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.groups.values()
        )


def _shard_groups(dataset) -> dict[int, list[int]]:
    shard_indices = getattr(dataset, "shard_indices", None)
    if shard_indices:
        groups: dict[int, list[int]] = {}
        for index, (shard_id, _) in enumerate(shard_indices):
            groups.setdefault(int(shard_id), []).append(index)
        return groups

    base_dataset = getattr(dataset, "dataset", None)
    subset_indices = getattr(dataset, "indices", None)
    base_shard_indices = getattr(base_dataset, "shard_indices", None)
    if base_dataset is not None and subset_indices is not None and base_shard_indices:
        groups = {}
        for subset_index, base_index in enumerate(subset_indices):
            shard_id, _ = base_shard_indices[int(base_index)]
            groups.setdefault(int(shard_id), []).append(subset_index)
        return groups
    return {}


def _make_train_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    groups = _shard_groups(dataset)
    if groups:
        return DataLoader(
            dataset,
            batch_sampler=ShardedBatchSampler(groups, batch_size, seed),
            num_workers=num_workers,
            collate_fn=collate_graphs,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_graphs,
    )


def evaluate(
    model,
    loader,
    device,
    pretraining,
    epoch: int,
    show_progress: bool,
    task_weighting: AdaptiveTaskWeighting | None = None,
) -> dict[str, float]:
    model.eval()
    loss_sums: dict[str, float] = {}
    metric_accumulator = PretrainingMetricAccumulator()
    sample_count = 0
    progress = _progress(
        loader,
        desc=f"epoch {epoch:03d} validation",
        enabled=show_progress,
    )
    with torch.no_grad():
        for graph in progress:
            graph = graph.to(device)
            output = model(
                graph,
                mask_ratio=1.0,
                random_replace_ratio=0.0,
                edge_noise_std=0.0,
                pseudo_mutation=pretraining.objective == "mutation_flow",
            )
            loss, parts = _loss_for_output(output, graph, pretraining)
            if task_weighting is not None:
                loss = task_weighting(parts)
                parts["total"] = loss
            batch_samples = graph.num_graphs
            sample_count += batch_samples
            _accumulate_losses(loss_sums, parts, batch_samples)
            metric_accumulator.update(output, graph)
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(loss=float(loss.detach().cpu().item()))
    losses = _average_losses(loss_sums, sample_count)
    losses.update(
        metric_accumulator.compute(losses.get("central_residue", 0.0))
    )
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pythia-style local structure pretraining"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--objective",
        choices=("pythia", "flow", "mutation_flow"),
        default=None,
    )
    args = parser.parse_args()
    config = load_config(args.config)
    pretraining = config.pretraining
    if args.objective is not None:
        pretraining.objective = args.objective
    data_path = args.data or pretraining.data_path
    if not data_path:
        raise ValueError(
            "Configure pretraining.data_path with a CATH/BioA graph pickle"
        )
    set_seed(config.seed)
    device = resolve_device(config.device)
    print(f"Using device: {describe_device(device)}", flush=True)

    dataset = StructureGraphDataset(
        data_path,
        deduplicate_proteins=pretraining.deduplicate_proteins,
        local_crop_radius=pretraining.local_crop_radius,
        local_crop_max_nodes=pretraining.local_crop_max_nodes,
        local_crop_neighbors=pretraining.local_crop_neighbors,
        centers_per_protein=pretraining.centers_per_protein,
        fixed_centers=False,
        center_seed=config.seed,
    )
    if pretraining.validation_path:
        train_dataset = dataset
        validation_dataset = StructureGraphDataset(
            pretraining.validation_path,
            deduplicate_proteins=pretraining.deduplicate_proteins,
            local_crop_radius=pretraining.local_crop_radius,
            local_crop_max_nodes=pretraining.local_crop_max_nodes,
            local_crop_neighbors=pretraining.local_crop_neighbors,
            centers_per_protein=pretraining.validation_centers_per_protein,
            fixed_centers=pretraining.fixed_validation_centers,
            center_seed=config.seed + 10_000,
        )
    else:
        validation_size = max(
            1, round(len(dataset) * pretraining.validation_fraction)
        )
        train_size = len(dataset) - validation_size
        if train_size < 1:
            raise ValueError("Structure pretraining needs at least two graphs")
        train_dataset, validation_dataset = random_split(
            dataset,
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(config.seed),
        )
    train_loader = _make_train_loader(
        train_dataset,
        batch_size=pretraining.batch_size,
        num_workers=config.data.num_workers,
        seed=config.seed,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=pretraining.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_graphs,
    )
    epochs = args.epochs or pretraining.epochs
    shard_groups = _shard_groups(train_dataset)
    sampler_name = "sharded" if shard_groups else "random"
    print(
        "Structure pretraining setup: "
        f"objective={pretraining.objective} "
        f"epochs={epochs} "
        f"train_graphs={len(train_dataset)} "
        f"validation_graphs={len(validation_dataset)} "
        f"train_batches={len(train_loader)} "
        f"validation_batches={len(validation_loader)} "
        f"batch_size={pretraining.batch_size} "
        f"sampler={sampler_name} "
        f"train_shards={len(shard_groups)} "
        f"node_dim={config.model.node_dim} "
        f"edge_dim={config.model.edge_dim} "
        f"local_crop_neighbors={pretraining.local_crop_neighbors} "
        f"centers_per_protein={pretraining.centers_per_protein} "
        f"fixed_validation_centers={pretraining.fixed_validation_centers} "
        f"checkpoint={pretraining.checkpoint_path}",
        flush=True,
    )
    if pretraining.objective in {"flow", "mutation_flow"}:
        model = StructureFlowPretrainingModel(
            node_dim=config.model.node_dim,
            edge_dim=config.model.edge_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_structure_layers,
            latent_dim=pretraining.latent_dim or config.model.latent_dim,
            flow_steps=config.model.flow_steps,
            flow_noise_scale=config.model.flow_noise_scale,
            dropout=config.model.dropout,
        ).to(device)
    else:
        model = StructurePretrainingModel(
            node_dim=config.model.node_dim,
            edge_dim=config.model.edge_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_structure_layers,
            dropout=config.model.dropout,
        ).to(device)
    task_weighting = None
    if (
        pretraining.adaptive_loss_weighting
        and pretraining.objective in {"flow", "mutation_flow"}
    ):
        task_priors = {
                "central_residue": pretraining.central_weight,
                "flow": pretraining.flow_weight,
                "transport": pretraining.transport_weight,
                "contrastive": pretraining.contrastive_weight,
                "geometry": pretraining.geometry_weight,
                "contact": pretraining.contact_weight,
        }
        if pretraining.objective == "mutation_flow":
            task_priors["replacement"] = pretraining.replacement_weight
        task_weighting = AdaptiveTaskWeighting(task_priors).to(device)
    optimized_parameters = list(model.parameters())
    if task_weighting is not None:
        optimized_parameters.extend(task_weighting.parameters())
    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    checkpoint_path = Path(pretraining.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sums: dict[str, float] = {}
        train_metric_accumulator = PretrainingMetricAccumulator()
        train_sample_count = 0
        progress = _progress(
            train_loader,
            desc=f"epoch {epoch:03d}/{epochs:03d} train",
            enabled=config.training.show_progress,
        )
        for graph in progress:
            graph = graph.to(device)
            output = model(
                graph,
                mask_ratio=pretraining.mask_ratio,
                random_replace_ratio=pretraining.random_replace_ratio,
                edge_noise_std=pretraining.edge_noise_std,
                pseudo_mutation=pretraining.objective == "mutation_flow",
            )
            loss, parts = _loss_for_output(output, graph, pretraining)
            if task_weighting is not None:
                loss = task_weighting(parts)
                parts["total"] = loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip
            )
            optimizer.step()
            batch_samples = graph.num_graphs
            train_sample_count += batch_samples
            _accumulate_losses(train_loss_sums, parts, batch_samples)
            train_metric_accumulator.update(output, graph)
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(loss=float(loss.detach().cpu().item()))
        train_losses = _average_losses(train_loss_sums, train_sample_count)
        train_losses.update(
            train_metric_accumulator.compute(
                train_losses.get("central_residue", 0.0)
            )
        )
        validation_losses = evaluate(
            model,
            validation_loader,
            device,
            pretraining,
            epoch=epoch,
            show_progress=config.training.show_progress,
            task_weighting=task_weighting,
        )
        validation_loss = validation_losses["total"]
        improved = validation_loss < best_validation
        print(
            f"epoch={epoch:03d} "
            f"{_format_losses('train', train_losses)} "
            f"{_format_losses('val', validation_losses)} "
            f"best_val={min(best_validation, validation_loss):.4f} "
            f"checkpoint_saved={str(improved).lower()}",
            (
                f" adaptive_weights={task_weighting.weights()}"
                if task_weighting is not None
                else ""
            ),
            flush=True,
        )
        if improved:
            best_validation = validation_loss
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state_dict": model.encoder.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "validation_loss": validation_loss,
                    "pretraining_objective": pretraining.objective,
                    "adaptive_task_weighting_state_dict": (
                        task_weighting.state_dict()
                        if task_weighting is not None
                        else None
                    ),
                    "config": config,
                },
                checkpoint_path,
            )


if __name__ == "__main__":
    main()
