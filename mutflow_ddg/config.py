from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExternalTestConfig:
    name: str
    path: str
    dataset_name: str | None = None
    manifest_path: str | None = None
    evaluation_scope: str = "full_set"
    esm_cache_dir: str | None = None
    esm_cache_fallback_dirs: list[str] = field(default_factory=list)
    esm_layout: str = "standard"
    pad_node_features: bool = False
    skip_missing_esm: bool = False
    batch_size: int = 32
    max_attention_elements: int | None = None
    ddg_multiplier: float = 1.0
    report_directions: bool = False

    def __post_init__(self) -> None:
        if self.evaluation_scope not in {"full_set", "common_subset"}:
            raise ValueError(
                "external test evaluation_scope must be full_set or common_subset"
            )
        valid_layouts = {"standard", "paired_flat"}
        if self.esm_layout not in valid_layouts:
            choices = ", ".join(sorted(valid_layouts))
            raise ValueError(
                f"esm_layout must be one of: {choices}; "
                f"got {self.esm_layout!r}"
            )
        if self.batch_size < 1:
            raise ValueError("external test batch_size must be positive")
        if (
            self.max_attention_elements is not None
            and self.max_attention_elements < 1
        ):
            raise ValueError(
                "external test max_attention_elements must be positive"
            )
        if self.ddg_multiplier == 0:
            raise ValueError("external test ddg_multiplier cannot be zero")


@dataclass
class DataConfig:
    train_dataset_name: str = "s2648"
    train_path: str | None = None
    train_manifest_path: str | None = None
    sample_set: str = "current"
    train_ablation_path: str | None = None
    validation_path: str | None = None
    validation_dataset_name: str | None = None
    validation_manifest_path: str | None = None
    esm_cache_dir: str | None = None
    esm_cache_fallback_dirs: list[str] = field(default_factory=list)
    validation_esm_cache_dir: str | None = None
    sequence_feature_source: str = "pssm"
    require_esm: bool = False
    batch_size: int = 32
    validation_fraction: float = 0.1
    split_level: str = "sample"
    add_reverse_mutations: bool = True
    num_workers: int = 0
    max_attention_elements: int | None = 4_000_000
    external_tests: list[ExternalTestConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        valid_sample_sets = {"current", "train_ablation_model"}
        if self.sample_set not in valid_sample_sets:
            choices = ", ".join(sorted(valid_sample_sets))
            raise ValueError(
                f"data.sample_set must be one of: {choices}; "
                f"got {self.sample_set!r}"
            )
        if (
            self.sample_set == "train_ablation_model"
            and not self.train_ablation_path
        ):
            raise ValueError(
                "data.train_ablation_path is required when "
                "data.sample_set=train_ablation_model"
            )
        valid_sequence_sources = {"pssm", "esm", "esm_pssm"}
        if self.sequence_feature_source not in valid_sequence_sources:
            choices = ", ".join(sorted(valid_sequence_sources))
            raise ValueError(
                f"data.sequence_feature_source must be one of: {choices}; "
                f"got {self.sequence_feature_source!r}"
            )
        valid_split_levels = {"sample", "protein"}
        if self.split_level not in valid_split_levels:
            choices = ", ".join(sorted(valid_split_levels))
            raise ValueError(
                f"data.split_level must be one of: {choices}; "
                f"got {self.split_level!r}"
            )
        if (
            self.max_attention_elements is not None
            and self.max_attention_elements < 1
        ):
            raise ValueError("data.max_attention_elements must be positive")


@dataclass
class ModelConfig:
    node_dim: int = 67
    sequence_dim: int = 41
    sequence_fusion: str = "linear"
    sequence_esm_dim: int = 1280
    sequence_pssm_dim: int = 41
    edge_dim: int = 16
    mutation_dim: int = 0
    mutation_feature_mode: str = "none"
    hidden_dim: int = 128
    latent_dim: int = 128
    num_structure_layers: int = 3
    num_sequence_layers: int = 2
    num_attention_heads: int = 8
    dropout: float = 0.1
    flow_steps: int = 8
    flow_noise_scale: float = 0.05
    num_classes: int = 3
    structure_checkpoint: str | None = None
    freeze_structure_encoder: bool = False
    # Legacy CoMDDG-compatible runs transferred the structure encoder only.
    # Flow transfer must be requested explicitly by a dedicated experiment.
    transfer_pretrained_flow: bool = False
    antisymmetric_output: bool = False
    branch_mode: str = "both"

    def __post_init__(self) -> None:
        valid_modes = {"sequence", "structure", "both"}
        if self.branch_mode not in valid_modes:
            choices = ", ".join(sorted(valid_modes))
            raise ValueError(
                f"model.branch_mode must be one of: {choices}; "
                f"got {self.branch_mode!r}"
            )
        valid_sequence_fusions = {"linear", "projected_gated"}
        if self.sequence_fusion not in valid_sequence_fusions:
            choices = ", ".join(sorted(valid_sequence_fusions))
            raise ValueError(
                f"model.sequence_fusion must be one of: {choices}; "
                f"got {self.sequence_fusion!r}"
            )
        if (
            self.sequence_fusion == "projected_gated"
            and self.sequence_dim
            != self.sequence_esm_dim + self.sequence_pssm_dim
        ):
            raise ValueError(
                "projected_gated sequence fusion requires sequence_dim = "
                "sequence_esm_dim + sequence_pssm_dim"
            )
        valid_mutation_features = {"none", "aa_properties"}
        if self.mutation_feature_mode not in valid_mutation_features:
            raise ValueError(
                "model.mutation_feature_mode must be none or aa_properties"
            )
        expected_mutation_dim = (
            44 if self.mutation_feature_mode == "aa_properties" else 0
        )
        if self.mutation_dim != expected_mutation_dim:
            raise ValueError(
                f"model.mutation_feature_mode={self.mutation_feature_mode} "
                f"requires mutation_dim={expected_mutation_dim}"
            )


@dataclass
class LossConfig:
    regression: float = 1.0
    classification: float = 0.2
    ranking: float = 0.1
    symmetry: float = 0.0
    transport: float = 0.1
    uncertainty: float = 0.1
    flow: float = 0.5
    epistasis: float = 0.0
    huber_delta: float = 1.0
    neutral_threshold: float = 0.5
    ddg_convention: str = "positive_destabilizing"


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 3e-4
    # Keep unspecified configs compatible with the original CoMDDG training:
    # all model parameters use the same learning rate, without warmup/freezing.
    structure_encoder_learning_rate: float = 3e-4
    new_layers_learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 0
    scheduler: str = "plateau"
    freeze_structure_epochs: int = 0
    gradual_unfreeze: bool = False
    unfreeze_layers_per_epoch: int = 1
    gradient_clip: float = 1.0
    patience: int = 20
    checkpoint_dir: str = "outputs/checkpoints"
    top_k_checkpoints: int = 5
    external_test_interval: int = 10
    show_progress: bool = True
    early_stopping_min_delta: float = 0.0
    require_protein_level_validation: bool = False

    def __post_init__(self) -> None:
        if self.top_k_checkpoints < 1:
            raise ValueError("training.top_k_checkpoints must be positive")
        if self.external_test_interval < 0:
            raise ValueError("training.external_test_interval cannot be negative")
        if self.structure_encoder_learning_rate <= 0:
            raise ValueError(
                "training.structure_encoder_learning_rate must be positive"
            )
        if self.new_layers_learning_rate <= 0:
            raise ValueError(
                "training.new_layers_learning_rate must be positive"
            )
        if self.scheduler not in {"warmup_cosine", "plateau"}:
            raise ValueError("training.scheduler must be warmup_cosine or plateau")
        if self.warmup_epochs < 0 or self.freeze_structure_epochs < 0:
            raise ValueError("warmup/freeze epochs cannot be negative")
        if self.unfreeze_layers_per_epoch < 1:
            raise ValueError("unfreeze_layers_per_epoch must be positive")


@dataclass
class PretrainingConfig:
    objective: str = "pythia"
    data_path: str | None = None
    validation_path: str | None = None
    batch_size: int = 32
    epochs: int = 50
    mask_ratio: float = 0.8
    random_replace_ratio: float = 0.1
    edge_noise_std: float = 0.5
    central_weight: float = 1.0
    flow_weight: float = 1.0
    transport_weight: float = 0.1
    contrastive_weight: float = 0.2
    replacement_weight: float = 0.5
    geometry_weight: float = 0.2
    contact_weight: float = 0.1
    adaptive_loss_weighting: bool = False
    contrastive_temperature: float = 0.1
    latent_dim: int | None = None
    validation_fraction: float = 0.05
    deduplicate_proteins: bool = True
    local_crop_radius: float | None = 10.0
    local_crop_max_nodes: int | None = 256
    local_crop_neighbors: int | None = None
    centers_per_protein: int = 4
    validation_centers_per_protein: int = 8
    fixed_validation_centers: bool = True
    checkpoint_path: str = "outputs/checkpoints/structure_pretrained.pt"

    def __post_init__(self) -> None:
        valid_objectives = {"pythia", "flow", "mutation_flow"}
        if self.objective not in valid_objectives:
            choices = ", ".join(sorted(valid_objectives))
            raise ValueError(
                f"pretraining.objective must be one of: {choices}; "
                f"got {self.objective!r}"
            )
        if self.local_crop_radius is not None and self.local_crop_radius <= 0:
            raise ValueError(
                "pretraining.local_crop_radius must be positive or null"
            )
        if (
            self.local_crop_max_nodes is not None
            and self.local_crop_max_nodes < 1
        ):
            raise ValueError(
                "pretraining.local_crop_max_nodes must be positive or null"
            )
        if (
            self.local_crop_neighbors is not None
            and self.local_crop_neighbors < 1
        ):
            raise ValueError(
                "pretraining.local_crop_neighbors must be positive or null"
            )
        if self.contrastive_weight < 0:
            raise ValueError("pretraining.contrastive_weight cannot be negative")
        if self.contrastive_temperature <= 0:
            raise ValueError(
                "pretraining.contrastive_temperature must be positive"
            )
        if not 4 <= self.centers_per_protein <= 8:
            raise ValueError("pretraining.centers_per_protein must be in [4, 8]")
        if self.validation_centers_per_protein < 1:
            raise ValueError(
                "pretraining.validation_centers_per_protein must be positive"
            )
        if self.mask_ratio + self.random_replace_ratio > 1:
            raise ValueError(
                "pretraining mask and random-replace ratios must sum to <= 1"
            )


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "cuda:0"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pretraining: PretrainingConfig = field(default_factory=PretrainingConfig)

    def __post_init__(self) -> None:
        combined = self.data.sequence_feature_source == "esm_pssm"
        gated = self.model.sequence_fusion == "projected_gated"
        if combined != gated:
            raise ValueError(
                "data.sequence_feature_source=esm_pssm and "
                "model.sequence_fusion=projected_gated must be enabled together"
            )


def _build_dataclass(cls: type, values: dict[str, Any] | None):
    return cls(**(values or {}))


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    data_values = dict(raw.get("data") or {})
    data_values["external_tests"] = [
        _build_dataclass(ExternalTestConfig, values)
        for values in data_values.get("external_tests", [])
    ]
    return ExperimentConfig(
        seed=raw.get("seed", 42),
        device=raw.get("device", "cuda:0"),
        data=_build_dataclass(DataConfig, data_values),
        model=_build_dataclass(ModelConfig, raw.get("model")),
        loss=_build_dataclass(LossConfig, raw.get("loss")),
        training=_build_dataclass(TrainingConfig, raw.get("training")),
        pretraining=_build_dataclass(PretrainingConfig, raw.get("pretraining")),
    )
