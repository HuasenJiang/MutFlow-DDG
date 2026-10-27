from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from mutflow_ddg.config import load_config
from mutflow_ddg.data import CoMDDGPairDataset, collate_graph_pairs
from mutflow_ddg.data.split import (
    ReverseAugmentedDataset,
    reverse_augmented_protein_split,
    reverse_augmented_random_split,
)
from mutflow_ddg.device import describe_device, resolve_device
from mutflow_ddg.engine import Trainer
from mutflow_ddg.models import MutFlowDDG


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_resume_path(
    requested_path: str | None,
    checkpoint_dir: str | Path,
) -> str | None:
    if requested_path != "auto":
        return requested_path
    candidates = [
        Path(checkpoint_dir) / "last.pt",
        Path(checkpoint_dir) / "best.pt",
    ]
    return next(
        (str(path) for path in candidates if path.is_file()),
        None,
    )


def validate_dataset_dimensions(dataset, config) -> None:
    sample = dataset[0]
    checks = {}
    if config.model.branch_mode in {"structure", "both"}:
        checks["model.node_dim"] = (
            sample.wild.x.shape[-1],
            config.model.node_dim,
        )
        checks["model.edge_dim"] = (
            sample.wild.edge_attr.shape[-1],
            config.model.edge_dim,
        )
    if config.model.branch_mode in {"sequence", "both"}:
        sequence_graph = sample.wild_for_sequence
        if sequence_graph.sequence_features is None:
            raise ValueError(
                "The selected branch mode requires sequence features"
            )
        checks["model.sequence_dim"] = (
            sequence_graph.sequence_features.shape[-1],
            config.model.sequence_dim,
        )
    errors = [
        f"{name}: data={actual}, config={expected}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if errors:
        raise ValueError("Feature dimension mismatch: " + "; ".join(errors))


def load_structure_checkpoint(
    model: MutFlowDDG,
    path: str,
    freeze: bool = False,
    transfer_flow: bool = True,
) -> dict[str, int]:
    if model.structure_encoder is None:
        raise ValueError(
            "model.structure_checkpoint cannot be used with branch_mode=sequence"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("encoder_state_dict", checkpoint)
    model.structure_encoder.load_state_dict(state)
    transferred = {"encoder": len(state), "flow": 0, "state_projection": 0}
    full_state = checkpoint.get("model_state_dict", {})
    if transfer_flow and full_state:
        flow_state = {
            key.removeprefix("flow."): value
            for key, value in full_state.items()
            if key.startswith("flow.")
        }
        if flow_state:
            model.flow.load_state_dict(flow_state, strict=True)
            transferred["flow"] = len(flow_state)
        if model.state_projection is not None:
            source_projection = {
                key.removeprefix("state_projection."): value
                for key, value in full_state.items()
                if key.startswith("state_projection.")
            }
            target_state = model.state_projection.state_dict()
            mapped = {}
            for key, value in source_projection.items():
                candidate = f"network.{key}"
                if (
                    candidate in target_state
                    and target_state[candidate].shape == value.shape
                ):
                    mapped[candidate] = value
            if mapped:
                model.state_projection.load_state_dict(mapped, strict=False)
                transferred["state_projection"] = len(mapped)
    if freeze:
        model.freeze_structure_encoder()
    print(f"Pretrained components transferred: {transferred}")
    return transferred


class AttentionBudgetBatchSampler(Sampler[list[int]]):
    """Group similarly sized proteins under a quadratic attention budget."""

    def __init__(
        self,
        lengths: list[int],
        max_batch_size: int,
        max_attention_elements: int,
        shuffle: bool = False,
        seed: int = 0,
    ):
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.batches: list[list[int]] = []
        batch: list[int] = []
        max_length = 0
        for index in sorted(range(len(lengths)), key=lengths.__getitem__):
            next_max_length = max(max_length, lengths[index])
            next_batch_size = len(batch) + 1
            exceeds_budget = (
                next_batch_size * next_max_length * next_max_length
                > max_attention_elements
            )
            if batch and (
                next_batch_size > max_batch_size or exceeds_budget
            ):
                self.batches.append(batch)
                batch = []
                max_length = 0
            batch.append(index)
            max_length = max(max_length, lengths[index])
        if batch:
            self.batches.append(batch)

    def __iter__(self):
        batches = [batch.copy() for batch in self.batches]
        if self.shuffle:
            generator = random.Random(self.seed + self.epoch)
            for batch in batches:
                generator.shuffle(batch)
            generator.shuffle(batches)
            self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self.batches)


def dataset_pair_length_at(dataset, index: int) -> int:
    if isinstance(dataset, Subset):
        return dataset_pair_length_at(dataset.dataset, dataset.indices[index])
    if hasattr(dataset, "pair_length_at"):
        return dataset.pair_length_at(index)
    sample = dataset[index]
    return max(
        sample.wild_for_sequence.x.shape[0],
        sample.mutant_for_sequence.x.shape[0],
    )


def dataset_pair_lengths(dataset) -> list[int]:
    return [
        dataset_pair_length_at(dataset, index)
        for index in range(len(dataset))
    ]


def build_loader(
    dataset,
    config,
    *,
    shuffle: bool,
    uses_sequence: bool,
    seed: int,
) -> DataLoader:
    if uses_sequence and config.data.max_attention_elements is not None:
        batch_sampler = AttentionBudgetBatchSampler(
            dataset_pair_lengths(dataset),
            max_batch_size=config.data.batch_size,
            max_attention_elements=config.data.max_attention_elements,
            shuffle=shuffle,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=config.data.num_workers,
            collate_fn=collate_graph_pairs,
        )
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        collate_fn=collate_graph_pairs,
    )


def build_external_test_loaders(config) -> dict[str, DataLoader]:
    loaders = {}
    uses_sequence = config.model.branch_mode in {"sequence", "both"}
    uses_structure = config.model.branch_mode in {"structure", "both"}
    uses_esm = uses_sequence and config.data.sequence_feature_source in {
        "esm",
        "esm_pssm",
    }
    for test_config in config.data.external_tests:
        if test_config.name in loaders:
            raise ValueError(
                f"Duplicate external test dataset name: {test_config.name!r}"
            )
        dataset = CoMDDGPairDataset(
            test_config.path,
            esm_cache_dir=(
                (
                    [test_config.esm_cache_dir, *test_config.esm_cache_fallback_dirs]
                    if test_config.esm_cache_fallback_dirs
                    else test_config.esm_cache_dir
                )
                if uses_esm
                else None
            ),
            require_esm=uses_esm and config.data.require_esm,
            esm_layout=test_config.esm_layout,
            sequence_feature_source=(
                config.data.sequence_feature_source
                if uses_sequence
                else "none"
            ),
            sequence_feature_dim=(
                config.model.sequence_dim if uses_sequence else None
            ),
            node_feature_dim=(
                config.model.node_dim
                if uses_structure and test_config.pad_node_features
                else None
            ),
            skip_missing_esm=test_config.skip_missing_esm,
            ddg_multiplier=test_config.ddg_multiplier,
            mutation_feature_mode=config.model.mutation_feature_mode,
            dataset_name=test_config.dataset_name or test_config.name,
            manifest_path=test_config.manifest_path,
        )
        validate_dataset_dimensions(dataset, config)
        batch_size = test_config.batch_size or config.data.batch_size
        if uses_sequence and test_config.max_attention_elements is not None:
            batch_sampler = AttentionBudgetBatchSampler(
                dataset_pair_lengths(dataset),
                max_batch_size=batch_size,
                max_attention_elements=test_config.max_attention_elements,
            )
            loaders[test_config.name] = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=config.data.num_workers,
                collate_fn=collate_graph_pairs,
            )
            batching_note = (
                f", {len(batch_sampler)} length-aware batches"
            )
        else:
            loaders[test_config.name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                collate_fn=collate_graph_pairs,
            )
            batching_note = ""
        loaders[test_config.name].report_directions = (
            test_config.report_directions
        )
        loaders[test_config.name].evaluation_scope = test_config.evaluation_scope
        padding_note = (
            f", node features padded to {config.model.node_dim}"
            if uses_structure and test_config.pad_node_features
            else ""
        )
        sequence_note = (
            f", {config.data.sequence_feature_source.upper()} features "
            f"{config.model.sequence_dim}-D"
            if uses_sequence
            else ""
        )
        ddg_note = (
            f", DDG x {test_config.ddg_multiplier:g}"
            if test_config.ddg_multiplier != 1
            else ""
        )
        direction_note = (
            ", Total/Direct/Reverse metrics"
            if test_config.report_directions
            else ""
        )
        coverage_note = (
            f"/{dataset.original_size} samples; "
            f"{dataset.skipped_missing_esm} skipped for missing ESM"
            if dataset.skipped_missing_esm
            else " samples"
        )
        print(
            f"External test loaded: {test_config.name} "
            f"({len(dataset)}{coverage_note}{sequence_note}"
            f"{padding_note}{ddg_note}{direction_note}{batching_note})"
        )
    return loaders


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MutFlow-DDG")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Resume from a checkpoint path. With no path, prefer last.pt and "
            "fall back to best.pt in the configured checkpoint directory."
        ),
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate retained Top-K checkpoints without retraining.",
    )
    args = parser.parse_args()
    if args.evaluate_only and args.resume is not None:
        parser.error("--evaluate-only and --resume cannot be used together")

    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
        config.training.checkpoint_dir = (
            f"{config.training.checkpoint_dir}_seed{args.seed}"
        )
    if config.data.sample_set == "current" and not config.data.train_path:
        raise ValueError("data.train_path must be configured")
    if (
        config.data.sample_set == "train_ablation_model"
        and config.model.branch_mode != "sequence"
    ):
        raise ValueError(
            "data.sample_set=train_ablation_model uses sequence graphs and "
            "requires "
            "model.branch_mode=sequence"
        )
    if (
        config.training.require_protein_level_validation
        and config.data.validation_path is None
        and config.data.split_level != "protein"
    ):
        raise ValueError(
            "This experiment requires protein-level validation for early "
            "stopping; set data.split_level=protein"
        )
    if (
        config.data.sample_set == "train_ablation_model"
        and config.data.validation_path
    ):
        raise ValueError(
            "train_ablation_model samples must use the reference random split; "
            "data.validation_path must be null"
        )
    set_seed(config.seed)
    device = resolve_device(config.device)
    print(f"Using device: {describe_device(device)}")

    uses_ablation_samples = (
        config.data.sample_set == "train_ablation_model"
    )
    train_path = (
        config.data.train_ablation_path
        if uses_ablation_samples
        else config.data.train_path
    )
    print(f"Training sample set: {config.data.sample_set}")
    print(f"Training sample path: {train_path}")
    uses_sequence = config.model.branch_mode in {"sequence", "both"}
    uses_esm = uses_sequence and config.data.sequence_feature_source in {
        "esm",
        "esm_pssm",
    }
    dataset = CoMDDGPairDataset(
        train_path,
        esm_cache_dir=(
            [config.data.esm_cache_dir, *config.data.esm_cache_fallback_dirs]
            if config.data.esm_cache_fallback_dirs
            else config.data.esm_cache_dir
        ) if uses_esm else None,
        require_esm=uses_esm and config.data.require_esm,
        sequence_feature_source=(
            config.data.sequence_feature_source if uses_sequence else "none"
        ),
        sequence_feature_dim=(
            config.model.sequence_dim if uses_sequence else None
        ),
        ignore_node_features=uses_ablation_samples,
        mutation_feature_mode=config.model.mutation_feature_mode,
        dataset_name=config.data.train_dataset_name,
        manifest_path=config.data.train_manifest_path,
    )
    validate_dataset_dimensions(dataset, config)
    sample = dataset[0]
    print(f"Branch mode: {config.model.branch_mode}")
    if config.model.branch_mode in {"sequence", "both"}:
        sequence_graph = sample.wild_for_sequence
        print(
            f"{config.data.sequence_feature_source.upper()} features dimension: "
            f"{sequence_graph.sequence_features.shape[1]}"
        )
    if config.model.branch_mode in {"structure", "both"}:
        print(f"Node features dimension: {sample.wild.x.shape[1]}")
        print(f"Edge features dimension: {sample.wild.edge_attr.shape[1]}")
    if config.data.validation_path:
        if not config.data.add_reverse_mutations:
            raise ValueError(
                "CoMDDG-compatible training requires add_reverse_mutations=true"
            )
        train_dataset = ReverseAugmentedDataset(dataset)
        validation_dataset = CoMDDGPairDataset(
            config.data.validation_path,
            esm_cache_dir=(
                [
                    config.data.validation_esm_cache_dir
                    or config.data.esm_cache_dir,
                    *config.data.esm_cache_fallback_dirs,
                ]
                if config.data.esm_cache_fallback_dirs
                else (
                    config.data.validation_esm_cache_dir
                    or config.data.esm_cache_dir
                )
            ) if uses_esm else None,
            require_esm=uses_esm and config.data.require_esm,
            sequence_feature_source=(
                config.data.sequence_feature_source
                if uses_sequence
                else "none"
            ),
            sequence_feature_dim=(
                config.model.sequence_dim if uses_sequence else None
            ),
            mutation_feature_mode=config.model.mutation_feature_mode,
            dataset_name=(
                config.data.validation_dataset_name
                or config.data.train_dataset_name
            ),
            manifest_path=config.data.validation_manifest_path,
        )
        validate_dataset_dimensions(validation_dataset, config)
        validation_dataset = ReverseAugmentedDataset(validation_dataset)
    else:
        if not config.data.add_reverse_mutations:
            raise ValueError(
                "CoMDDG-compatible training requires add_reverse_mutations=true"
            )
        if config.data.split_level == "sample":
            train_dataset, validation_dataset = reverse_augmented_random_split(
                dataset,
                validation_fraction=config.data.validation_fraction,
                seed=config.seed,
            )
            print(
                "Data split: sample-level after reverse augmentation "
                "(CoMDDG-compatible)"
            )
        else:
            train_dataset, validation_dataset = (
                reverse_augmented_protein_split(
                    dataset,
                    validation_fraction=config.data.validation_fraction,
                    seed=config.seed,
                )
            )
            print(
                "Data split: protein-level before reverse augmentation "
                "(protein-disjoint)"
            )

    train_loader = build_loader(
        train_dataset,
        shuffle=True,
        uses_sequence=uses_sequence,
        config=config,
        seed=config.seed,
    )
    validation_loader = build_loader(
        validation_dataset,
        shuffle=False,
        uses_sequence=uses_sequence,
        config=config,
        seed=config.seed,
    )
    if uses_sequence and config.data.max_attention_elements is not None:
        print(
            "Length-aware batching enabled: "
            f"{len(train_loader)} train / {len(validation_loader)} validation "
            "batches"
        )
    external_test_loaders = build_external_test_loaders(config)

    model = MutFlowDDG(config.model)
    if config.model.structure_checkpoint and not args.evaluate_only:
        load_structure_checkpoint(
            model,
            config.model.structure_checkpoint,
            freeze=config.model.freeze_structure_encoder,
            transfer_flow=config.model.transfer_pretrained_flow,
        )
    elif config.model.freeze_structure_encoder and not args.evaluate_only:
        raise ValueError(
            "model.freeze_structure_encoder=true requires "
            "model.structure_checkpoint"
        )
    trainer = Trainer(model, config, device)
    if args.evaluate_only:
        trainer.evaluate_saved_checkpoints(
            validation_loader,
            external_test_loaders,
        )
        return
    resume_path = resolve_resume_path(
        args.resume,
        config.training.checkpoint_dir,
    )
    if args.resume == "auto" and resume_path is None:
        print("No checkpoint found; starting training from epoch 1")
    trainer.fit(
        train_loader,
        validation_loader,
        external_test_loaders=external_test_loaders,
        resume_path=resume_path,
    )


if __name__ == "__main__":
    main()
