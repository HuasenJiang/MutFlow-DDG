from __future__ import annotations

import csv
from collections import defaultdict
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .data.schema import MutationBatch
from .losses import MutFlowObjective
from .metrics import regression_metrics


def atomic_torch_save(payload, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_torch_checkpoint(
    path: Path,
    map_location: torch.device | str,
) -> dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_copy(source: Path, destination: Path) -> None:
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary_path)
    temporary_path.replace(destination)


def progress_bars_enabled(config: ExperimentConfig) -> bool:
    override = os.environ.get("MUTFLOW_PROGRESS")
    if override is None:
        return config.training.show_progress
    normalized = override.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "MUTFLOW_PROGRESS must be one of: 1/0, true/false, yes/no, on/off"
    )


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.objective = MutFlowObjective(config.loss)
        structure_parameters = []
        new_parameters = []
        for name, parameter in model.named_parameters():
            if name.startswith("structure_encoder."):
                structure_parameters.append(parameter)
            else:
                new_parameters.append(parameter)
        parameter_groups = []
        if structure_parameters:
            parameter_groups.append(
                {
                    "params": structure_parameters,
                    "lr": config.training.structure_encoder_learning_rate,
                    "group_name": "structure_encoder",
                }
            )
        if new_parameters:
            parameter_groups.append(
                {
                    "params": new_parameters,
                    "lr": config.training.new_layers_learning_rate,
                    "group_name": "new_layers",
                }
            )
        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=config.training.weight_decay,
        )
        if config.training.scheduler == "warmup_cosine":
            warmup = config.training.warmup_epochs
            total = max(config.training.epochs, warmup + 1)

            def lr_factor(epoch: int) -> float:
                completed = epoch + 1
                if warmup > 0 and completed <= warmup:
                    return completed / warmup
                progress = (completed - warmup) / max(total - warmup, 1)
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lr_factor
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.8,
                patience=max(config.training.patience // 2, 1),
                min_lr=1e-6,
                threshold=config.training.early_stopping_min_delta,
            )
        self._last_trainable_structure_layers: int | None = None
        if (
            hasattr(self.model, "set_structure_trainable_layers")
            and getattr(self.model, "structure_encoder", None) is not None
            and not config.model.freeze_structure_encoder
            and config.training.freeze_structure_epochs > 0
        ):
            self.model.set_structure_trainable_layers(0)
            self._last_trainable_structure_layers = 0

    def _update_structure_schedule(self, epoch: int) -> None:
        encoder = getattr(self.model, "structure_encoder", None)
        if (
            encoder is None
            or self.config.model.freeze_structure_encoder
            or not hasattr(self.model, "set_structure_trainable_layers")
        ):
            return
        total = len(encoder.layers)
        frozen_epochs = self.config.training.freeze_structure_epochs
        if epoch <= frozen_epochs:
            count = 0
        elif self.config.training.gradual_unfreeze:
            count = min(
                total,
                (epoch - frozen_epochs)
                * self.config.training.unfreeze_layers_per_epoch,
            )
        else:
            count = total
        if count != self._last_trainable_structure_layers:
            self.model.set_structure_trainable_layers(count)
            self._last_trainable_structure_layers = count
            print(
                "Structure encoder schedule: "
                f"epoch={epoch}, trainable_message_layers={count}/{total}"
            )

    @staticmethod
    def _regression_result(
        prediction: Tensor,
        target: Tensor,
    ) -> dict[str, float]:
        return {
            "samples": int(target.numel()),
            **regression_metrics(prediction, target),
        }

    def _collect_regression_outputs(
        self,
        loader,
        epoch: int,
        dataset_name: str,
    ) -> tuple[Tensor, Tensor, list[str]]:
        self.model.eval()
        predictions, targets = [], []
        mutation_ids: list[str] = []
        progress = tqdm(
            loader,
            desc=f"epoch {epoch:03d} predict {dataset_name}",
            dynamic_ncols=True,
            leave=False,
            disable=not progress_bars_enabled(self.config),
        )
        with torch.inference_mode():
            for batch in progress:
                batch = batch.to(self.device)
                output = self.model(batch)
                predictions.append(output["ddg"].detach().cpu())
                targets.append(batch.ddg.detach().cpu())
                if batch.mutation_ids is not None:
                    mutation_ids.extend(batch.mutation_ids)
                else:
                    mutation_ids.extend(
                        f"{dataset_name}:legacy:{len(mutation_ids) + index}"
                        for index in range(batch.ddg.numel())
                    )
                del output, batch
        if not predictions:
            raise ValueError(f"Prediction dataset {dataset_name!r} is empty")
        return torch.cat(predictions), torch.cat(targets), mutation_ids

    def _collect_directional_outputs(
        self,
        loader,
        epoch: int,
        dataset_name: str,
    ) -> dict[str, tuple[Tensor, Tensor, list[str]]]:
        self.model.eval()
        direct_predictions, direct_targets = [], []
        reverse_predictions, reverse_targets = [], []
        mutation_ids: list[str] = []
        progress = tqdm(
            loader,
            desc=f"epoch {epoch:03d} predict {dataset_name} direct+reverse",
            dynamic_ncols=True,
            leave=False,
            disable=not progress_bars_enabled(self.config),
        )
        with torch.inference_mode():
            for batch in progress:
                batch = batch.to(self.device)
                direct_output = self.model(batch)
                direct_predictions.append(
                    direct_output["ddg"].detach().cpu()
                )
                direct_targets.append(batch.ddg.detach().cpu())
                if batch.mutation_ids is not None:
                    mutation_ids.extend(batch.mutation_ids)
                else:
                    mutation_ids.extend(
                        f"{dataset_name}:legacy:{len(mutation_ids) + index}"
                        for index in range(batch.ddg.numel())
                    )
                del direct_output

                reverse_batch = batch.reversed()
                reverse_output = self.model(reverse_batch)
                reverse_predictions.append(
                    reverse_output["ddg"].detach().cpu()
                )
                reverse_targets.append(reverse_batch.ddg.detach().cpu())
                del reverse_output, reverse_batch, batch

        if not direct_predictions:
            raise ValueError(f"Prediction dataset {dataset_name!r} is empty")
        return {
            "direct": (
                torch.cat(direct_predictions),
                torch.cat(direct_targets),
                mutation_ids,
            ),
            "reverse": (
                torch.cat(reverse_predictions),
                torch.cat(reverse_targets),
                mutation_ids,
            ),
        }

    def _step(
        self, batch: MutationBatch, training: bool
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch = batch.to(self.device)
        output = self.model(batch)
        reverse_output = None
        if training and self.config.loss.symmetry > 0:
            reverse_output = self.model(batch.reversed())
        total, losses = self.objective(output, batch, reverse_output)
        if training:
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.gradient_clip
            )
            self.optimizer.step()
        return output, {**losses, "target": batch.ddg}

    def run_epoch(
        self,
        loader,
        training: bool,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train(training)
        totals = defaultdict(float)
        predictions, targets = [], []
        phase = "train" if training else "validation"
        progress = tqdm(
            loader,
            desc=f"epoch {epoch:03d} {phase}",
            dynamic_ncols=True,
            leave=False,
            disable=not progress_bars_enabled(self.config),
        )
        with torch.set_grad_enabled(training):
            for step, batch in enumerate(progress, start=1):
                output, state = self._step(batch, training)
                for name, value in state.items():
                    if name == "target":
                        continue
                    totals[name] += float(value.detach().cpu())
                predictions.append(output["ddg"].detach().cpu())
                targets.append(state["target"].detach().cpu())
                totals["l1"] += float(
                    F.l1_loss(output["ddg"], state["target"]).detach().cpu()
                )
                progress.set_postfix(
                    loss=f"{totals['total'] / step:.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )
                del output, state, batch

        result = {
            name: value / max(len(loader), 1) for name, value in totals.items()
        }
        direct = torch.cat(predictions)
        target = torch.cat(targets)
        result.update(regression_metrics(direct, target))
        return result

    def evaluate_regression(
        self,
        loader,
        epoch: int,
        dataset_name: str,
    ) -> dict[str, float]:
        prediction, target, _ = self._collect_regression_outputs(
            loader,
            epoch,
            dataset_name,
        )
        return self._regression_result(prediction, target)

    def evaluate_directional_regression(
        self,
        loader,
        epoch: int,
        dataset_name: str,
    ) -> dict[str, dict[str, float]]:
        outputs = self._collect_directional_outputs(
            loader,
            epoch,
            dataset_name,
        )
        direct_prediction, direct_target, _ = outputs["direct"]
        reverse_prediction, reverse_target, _ = outputs["reverse"]
        total_prediction = torch.cat(
            [direct_prediction, reverse_prediction]
        )
        total_target = torch.cat([direct_target, reverse_target])

        return {
            "total": self._regression_result(total_prediction, total_target),
            "direct": self._regression_result(
                direct_prediction,
                direct_target,
            ),
            "reverse": self._regression_result(
                reverse_prediction,
                reverse_target,
            ),
        }

    def evaluate_external_loaders(
        self,
        external_test_loaders,
        epoch: int,
    ) -> dict[str, dict[str, float]]:
        external_metrics = {}
        for name, loader in external_test_loaders.items():
            if getattr(loader, "report_directions", False):
                directional_metrics = self.evaluate_directional_regression(
                    loader, epoch, name
                )
                for direction, metrics in directional_metrics.items():
                    metrics["evaluation_scope"] = getattr(
                        loader, "evaluation_scope", "full_set"
                    )
                    external_metrics[f"{name}_{direction}"] = metrics
            else:
                external_metrics[name] = self.evaluate_regression(
                    loader, epoch, name
                )
                external_metrics[name]["evaluation_scope"] = getattr(
                    loader, "evaluation_scope", "full_set"
                )
        return external_metrics

    @staticmethod
    def write_external_metrics(
        path: Path,
        fieldnames: list[str],
        epoch: int,
        external_metrics: dict[str, dict[str, float]],
    ) -> None:
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            for name, metrics in external_metrics.items():
                writer.writerow(
                    {"epoch": epoch, "dataset": name, **metrics}
                )
                print(
                    f"External Test [{name}] "
                    f"N: {metrics.get('samples', 'unknown')}, "
                    f"MAE: {metrics['mae']:.4f}, "
                    f"RMSE: {metrics['rmse']:.4f}, "
                    f"PCC: {metrics['pcc']:.4f}"
                )

    def _checkpoint_payload(
        self,
        epoch: int,
        validation_metrics: dict[str, float],
        best_loss: float,
        best_metrics: dict[str, float],
        external_metrics: dict[str, dict[str, float]],
        top_checkpoints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": float(validation_metrics["l1"]),
            "best_loss": best_loss,
            "monitor": "l1",
            "metrics": validation_metrics,
            "best_metrics": best_metrics,
            "external_tests": external_metrics,
            "top_checkpoints": top_checkpoints,
            "config": self.config,
        }

    @staticmethod
    def _top_checkpoint_filename(epoch: int, loss: float) -> str:
        return f"top_epoch_{epoch:04d}_val_{loss:.8f}.pt"

    def _restore_top_checkpoints(
        self,
        checkpoint_dir: Path,
        resume_checkpoint: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if resume_checkpoint is None:
            for path in checkpoint_dir.glob("top_epoch_*_val_*.pt"):
                path.unlink(missing_ok=True)
            for name in (
                "top5_checkpoints.csv",
                "top5_metrics.csv",
                "top5_predictions.csv",
            ):
                (checkpoint_dir / name).unlink(missing_ok=True)
            return []

        entries: dict[tuple[int, float], dict[str, Any]] = {}

        def add_entry(epoch: int, loss: float, path: Path) -> None:
            if path.is_file():
                entries[(epoch, loss)] = {
                    "epoch": epoch,
                    "loss": loss,
                    "path": path.name,
                }

        if resume_checkpoint is not None:
            for entry in resume_checkpoint.get("top_checkpoints", []):
                path = checkpoint_dir / str(entry["path"])
                add_entry(
                    int(entry["epoch"]),
                    float(entry["loss"]),
                    path,
                )

        for path in checkpoint_dir.glob("top_epoch_*_val_*.pt"):
            checkpoint = load_torch_checkpoint(path, "cpu")
            add_entry(
                int(checkpoint["epoch"]),
                float(checkpoint.get("loss", checkpoint["metrics"]["l1"])),
                path,
            )

        legacy_best_path = checkpoint_dir / "best.pt"
        if not entries and legacy_best_path.is_file():
            checkpoint = load_torch_checkpoint(legacy_best_path, "cpu")
            epoch = int(checkpoint["epoch"])
            loss = float(
                checkpoint.get(
                    "loss",
                    checkpoint.get("best_loss", checkpoint["metrics"]["l1"]),
                )
            )
            archive_path = checkpoint_dir / self._top_checkpoint_filename(
                epoch,
                loss,
            )
            if not archive_path.is_file():
                atomic_copy(legacy_best_path, archive_path)
            add_entry(epoch, loss, archive_path)

        ranked = sorted(
            entries.values(),
            key=lambda entry: (entry["loss"], entry["epoch"]),
        )
        keep = ranked[: self.config.training.top_k_checkpoints]
        keep_paths = {entry["path"] for entry in keep}
        for entry in ranked[self.config.training.top_k_checkpoints :]:
            path = checkpoint_dir / entry["path"]
            if (
                path.name.startswith("top_epoch_")
                and path.name not in keep_paths
            ):
                path.unlink(missing_ok=True)
        if keep:
            atomic_copy(checkpoint_dir / keep[0]["path"], legacy_best_path)
        return keep

    def _save_top_candidate(
        self,
        checkpoint_dir: Path,
        checkpoint: dict[str, Any],
        top_checkpoints: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        epoch = int(checkpoint["epoch"])
        loss = float(checkpoint["metrics"]["l1"])
        candidate = {
            "epoch": epoch,
            "loss": loss,
            "path": self._top_checkpoint_filename(epoch, loss),
        }
        retained = [
            entry
            for entry in top_checkpoints
            if int(entry["epoch"]) != epoch
        ]
        ranked = sorted(
            [*retained, candidate],
            key=lambda entry: (entry["loss"], entry["epoch"]),
        )
        kept = ranked[: self.config.training.top_k_checkpoints]
        if candidate not in kept:
            return top_checkpoints

        checkpoint["top_checkpoints"] = kept
        candidate_path = checkpoint_dir / candidate["path"]
        atomic_torch_save(checkpoint, candidate_path)

        kept_paths = {entry["path"] for entry in kept}
        for entry in top_checkpoints:
            path = checkpoint_dir / entry["path"]
            if (
                entry["path"] not in kept_paths
                and path.name.startswith("top_epoch_")
            ):
                path.unlink(missing_ok=True)

        rank = kept.index(candidate) + 1
        best_path = checkpoint_dir / "best.pt"
        if rank == 1 or not best_path.is_file():
            atomic_copy(checkpoint_dir / kept[0]["path"], best_path)
        print(
            f"Top-{self.config.training.top_k_checkpoints} model saved: "
            f"rank {rank}, epoch {epoch}, validation L1 {loss:.4f}"
        )
        return kept

    @staticmethod
    def _write_top_checkpoint_manifest(
        path: Path,
        top_checkpoints: list[dict[str, Any]],
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["rank", "checkpoint", "epoch", "validation_l1"],
            )
            writer.writeheader()
            for rank, entry in enumerate(top_checkpoints, start=1):
                writer.writerow(
                    {
                        "rank": rank,
                        "checkpoint": entry["path"],
                        "epoch": entry["epoch"],
                        "validation_l1": entry["loss"],
                    }
                )

    @staticmethod
    def _write_prediction_rows(
        writer: csv.DictWriter,
        *,
        rank: int,
        checkpoint_name: str,
        epoch: int,
        dataset: str,
        direction: str,
        prediction: Tensor,
        target: Tensor,
        mutation_ids: list[str],
    ) -> None:
        for index, (mutation_id, predicted, expected) in enumerate(
            zip(mutation_ids, prediction.tolist(), target.tolist())
        ):
            writer.writerow(
                {
                    "rank": rank,
                    "checkpoint": checkpoint_name,
                    "epoch": epoch,
                    "dataset": dataset,
                    "direction": direction,
                    "sample_index": index,
                    "mutation_id": mutation_id,
                    "target": expected,
                    "prediction": predicted,
                }
            )

    def _evaluate_top_checkpoints(
        self,
        checkpoint_dir: Path,
        top_checkpoints: list[dict[str, Any]],
        validation_loader,
        external_test_loaders,
    ) -> dict[str, float]:
        metrics_path = checkpoint_dir / "top5_metrics.csv"
        predictions_path = checkpoint_dir / "top5_predictions.csv"
        metrics_fieldnames = [
            "rank",
            "checkpoint",
            "epoch",
            "dataset",
            "evaluation_scope",
            "samples",
            "mae",
            "mse",
            "rmse",
            "pcc",
            "spearman",
        ]
        prediction_fieldnames = [
            "rank",
            "checkpoint",
            "epoch",
            "dataset",
            "direction",
            "sample_index",
            "mutation_id",
            "target",
            "prediction",
        ]
        best_validation_metrics: dict[str, float] = {}

        with (
            metrics_path.open("w", newline="", encoding="utf-8") as metrics_file,
            predictions_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as predictions_file,
        ):
            metrics_writer = csv.DictWriter(
                metrics_file,
                fieldnames=metrics_fieldnames,
            )
            prediction_writer = csv.DictWriter(
                predictions_file,
                fieldnames=prediction_fieldnames,
            )
            metrics_writer.writeheader()
            prediction_writer.writeheader()

            for rank, entry in enumerate(top_checkpoints, start=1):
                checkpoint_path = checkpoint_dir / entry["path"]
                checkpoint = load_torch_checkpoint(
                    checkpoint_path,
                    self.device,
                )
                epoch = int(checkpoint["epoch"])
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(
                    f"\nPredicting with Top {rank}: {entry['path']} "
                    f"(epoch {epoch}, validation L1 {entry['loss']:.4f})"
                )

                validation_prediction, validation_target, validation_ids = (
                    self._collect_regression_outputs(
                        validation_loader,
                        epoch,
                        "validation",
                    )
                )
                validation_metrics = self._regression_result(
                    validation_prediction,
                    validation_target,
                )
                validation_metrics["l1"] = validation_metrics["mae"]
                metrics_writer.writerow(
                    {
                        "rank": rank,
                        "checkpoint": entry["path"],
                        "epoch": epoch,
                        "dataset": "validation",
                        "evaluation_scope": "protein_validation",
                        **{
                            name: validation_metrics[name]
                            for name in metrics_fieldnames
                            if name in validation_metrics
                        },
                    }
                )
                self._write_prediction_rows(
                    prediction_writer,
                    rank=rank,
                    checkpoint_name=entry["path"],
                    epoch=epoch,
                    dataset="validation",
                    direction="direct",
                    prediction=validation_prediction,
                    target=validation_target,
                    mutation_ids=validation_ids,
                )

                external_metrics: dict[str, dict[str, float]] = {}
                for name, loader in external_test_loaders.items():
                    if getattr(loader, "report_directions", False):
                        outputs = self._collect_directional_outputs(
                            loader,
                            epoch,
                            name,
                        )
                        (
                            direct_prediction,
                            direct_target,
                            direct_ids,
                        ) = outputs["direct"]
                        (
                            reverse_prediction,
                            reverse_target,
                            reverse_ids,
                        ) = outputs["reverse"]
                        metric_sets = {
                            f"{name}_total": self._regression_result(
                                torch.cat(
                                    [direct_prediction, reverse_prediction]
                                ),
                                torch.cat([direct_target, reverse_target]),
                            ),
                            f"{name}_direct": self._regression_result(
                                direct_prediction,
                                direct_target,
                            ),
                            f"{name}_reverse": self._regression_result(
                                reverse_prediction,
                                reverse_target,
                            ),
                        }
                        for metrics in metric_sets.values():
                            metrics["evaluation_scope"] = getattr(
                                loader, "evaluation_scope", "full_set"
                            )
                        external_metrics.update(metric_sets)
                        self._write_prediction_rows(
                            prediction_writer,
                            rank=rank,
                            checkpoint_name=entry["path"],
                            epoch=epoch,
                            dataset=name,
                            direction="direct",
                            prediction=direct_prediction,
                            target=direct_target,
                            mutation_ids=direct_ids,
                        )
                        self._write_prediction_rows(
                            prediction_writer,
                            rank=rank,
                            checkpoint_name=entry["path"],
                            epoch=epoch,
                            dataset=name,
                            direction="reverse",
                            prediction=reverse_prediction,
                            target=reverse_target,
                            mutation_ids=reverse_ids,
                        )
                    else:
                        prediction, target, mutation_ids = (
                            self._collect_regression_outputs(
                            loader,
                            epoch,
                            name,
                        )
                        )
                        external_metrics[name] = self._regression_result(
                            prediction,
                            target,
                        )
                        external_metrics[name]["evaluation_scope"] = getattr(
                            loader, "evaluation_scope", "full_set"
                        )
                        self._write_prediction_rows(
                            prediction_writer,
                            rank=rank,
                            checkpoint_name=entry["path"],
                            epoch=epoch,
                            dataset=name,
                            direction="direct",
                            prediction=prediction,
                            target=target,
                            mutation_ids=mutation_ids,
                        )

                for name, metrics in external_metrics.items():
                    metrics_writer.writerow(
                        {
                            "rank": rank,
                            "checkpoint": entry["path"],
                            "epoch": epoch,
                            "dataset": name,
                            **metrics,
                        }
                    )
                    print(
                        f"Top {rank} [{name}] N: {metrics['samples']}, "
                        f"MAE: {metrics['mae']:.4f}, "
                        f"RMSE: {metrics['rmse']:.4f}, "
                        f"PCC: {metrics['pcc']:.4f}"
                    )

                checkpoint["final_validation_metrics"] = validation_metrics
                checkpoint["external_tests"] = external_metrics
                checkpoint["top_rank"] = rank
                atomic_torch_save(checkpoint, checkpoint_path)
                if rank == 1:
                    atomic_copy(checkpoint_path, checkpoint_dir / "best.pt")
                    best_validation_metrics = validation_metrics

        return best_validation_metrics

    def evaluate_saved_checkpoints(
        self,
        validation_loader,
        external_test_loaders=None,
    ) -> dict[str, float]:
        """Evaluate retained Top-K models without entering the training loop."""
        external_test_loaders = external_test_loaders or {}
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        last_path = checkpoint_dir / "last.pt"
        resume_checkpoint = (
            load_torch_checkpoint(last_path, "cpu")
            if last_path.is_file()
            else None
        )
        if resume_checkpoint is None:
            best_path = checkpoint_dir / "best.pt"
            if best_path.is_file():
                resume_checkpoint = load_torch_checkpoint(best_path, "cpu")
        if resume_checkpoint is None:
            raise FileNotFoundError(
                f"No saved checkpoint found under {checkpoint_dir}"
            )
        top_checkpoints = self._restore_top_checkpoints(
            checkpoint_dir,
            resume_checkpoint,
        )
        if not top_checkpoints:
            raise FileNotFoundError(
                f"No retained Top-K checkpoint found under {checkpoint_dir}"
            )
        return self._evaluate_top_checkpoints(
            checkpoint_dir,
            top_checkpoints,
            validation_loader,
            external_test_loaders,
        )

    def fit(
        self,
        train_loader,
        validation_loader,
        external_test_loaders=None,
        resume_path: str | Path | None = None,
    ) -> dict[str, float]:
        external_test_loaders = external_test_loaders or {}
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        last_checkpoint_path = checkpoint_dir / "last.pt"
        external_metrics_path = checkpoint_dir / "external_test_metrics.csv"
        top_manifest_path = checkpoint_dir / "top5_checkpoints.csv"
        external_fieldnames = [
            "epoch",
            "dataset",
            "evaluation_scope",
            "samples",
            "mae",
            "mse",
            "rmse",
            "pcc",
            "spearman",
        ]
        best_loss = float("inf")
        best_metrics: dict[str, float] = {}
        epochs_without_improvement = 0
        start_epoch = 1
        resume_checkpoint = None
        if resume_path is not None:
            resume_path = Path(resume_path)
            if not resume_path.is_file():
                raise FileNotFoundError(
                    f"Resume checkpoint does not exist: {resume_path}"
                )
            resume_checkpoint = load_torch_checkpoint(
                resume_path,
                self.device,
            )
            self.model.load_state_dict(resume_checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(
                resume_checkpoint["optimizer_state_dict"]
            )
            if "scheduler_state_dict" in resume_checkpoint:
                self.scheduler.load_state_dict(
                    resume_checkpoint["scheduler_state_dict"]
                )
            start_epoch = int(resume_checkpoint["epoch"]) + 1
            best_loss = float(
                resume_checkpoint.get(
                    "best_loss",
                    resume_checkpoint.get("loss", float("inf")),
                )
            )
            best_metrics = dict(
                resume_checkpoint.get(
                    "best_metrics",
                    resume_checkpoint.get("metrics", {}),
                )
            )
            print(
                f"Resumed from {resume_path} at epoch "
                f"{start_epoch} (best validation L1: {best_loss:.4f})"
            )

        top_checkpoints = self._restore_top_checkpoints(
            checkpoint_dir,
            resume_checkpoint,
        )
        if top_checkpoints:
            best_loss = min(best_loss, float(top_checkpoints[0]["loss"]))

        if external_test_loaders and (
            start_epoch == 1 or not external_metrics_path.is_file()
        ):
            with external_metrics_path.open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                csv.DictWriter(handle, fieldnames=external_fieldnames).writeheader()

        print("Starting training for MutFlow-DDG model...")
        for epoch in range(start_epoch, self.config.training.epochs + 1):
            self._update_structure_schedule(epoch)
            start_time = time.time()
            train_metrics = self.run_epoch(
                train_loader, training=True, epoch=epoch
            )
            validation_metrics = self.run_epoch(
                validation_loader, training=False, epoch=epoch
            )
            interval = self.config.training.external_test_interval
            external_metrics = (
                self.evaluate_external_loaders(external_test_loaders, epoch)
                if interval > 0 and epoch % interval == 0
                else {}
            )
            if self.config.training.scheduler == "plateau":
                self.scheduler.step(validation_metrics["l1"])
            else:
                self.scheduler.step()
            current_lrs = {
                group.get("group_name", f"group_{index}"): group["lr"]
                for index, group in enumerate(self.optimizer.param_groups)
            }
            epoch_duration = time.time() - start_time
            print(
                f"Epoch [{epoch}/{self.config.training.epochs}] "
                f"Train Loss: {train_metrics['l1']:.4f}, "
                f"Val Loss: {validation_metrics['l1']:.4f}, "
                f"Val MAE: {validation_metrics['mae']:.4f}, "
                f"Val MSE: {validation_metrics['mse']:.4f}, "
                f"Val PCC: {validation_metrics['pcc']:.4f}, "
                f"LRs: {current_lrs}, "
                f"Time: {epoch_duration:.2f}s"
            )
            if external_metrics:
                self.write_external_metrics(
                    external_metrics_path,
                    external_fieldnames,
                    epoch,
                    external_metrics,
                )
            improved = (
                validation_metrics["l1"]
                < best_loss - self.config.training.early_stopping_min_delta
            )
            if improved:
                best_loss = validation_metrics["l1"]
                best_metrics = validation_metrics
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            checkpoint = self._checkpoint_payload(
                epoch,
                validation_metrics,
                best_loss,
                best_metrics,
                external_metrics,
                top_checkpoints,
            )
            top_checkpoints = self._save_top_candidate(
                checkpoint_dir,
                checkpoint,
                top_checkpoints,
            )
            last_checkpoint = self._checkpoint_payload(
                epoch,
                validation_metrics,
                best_loss,
                best_metrics,
                external_metrics,
                top_checkpoints,
            )
            atomic_torch_save(last_checkpoint, last_checkpoint_path)
            if epochs_without_improvement >= self.config.training.patience:
                print(
                    "Early stopping: protein-level validation L1 did not "
                    f"improve for {epochs_without_improvement} epochs."
                )
                break

        print("Training completed!")
        if not top_checkpoints:
            raise RuntimeError("Training produced no Top-K checkpoints")
        self._write_top_checkpoint_manifest(
            top_manifest_path,
            top_checkpoints,
        )
        print(
            f"\nPredicting with {len(top_checkpoints)} retained Top-K models..."
        )
        best_metrics = self._evaluate_top_checkpoints(
            checkpoint_dir,
            top_checkpoints,
            validation_loader,
            external_test_loaders,
        )
        return best_metrics
