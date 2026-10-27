from pathlib import Path

import pytest

from mutflow_ddg.config import load_config


def test_unspecified_training_options_keep_legacy_comddg_behavior(
    tmp_path: Path,
):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")

    config = load_config(path)

    assert config.model.transfer_pretrained_flow is False
    assert config.training.structure_encoder_learning_rate == 3e-4
    assert config.training.new_layers_learning_rate == 3e-4
    assert config.training.weight_decay == 1e-5
    assert config.training.scheduler == "plateau"
    assert config.training.warmup_epochs == 0
    assert config.training.freeze_structure_epochs == 0
    assert config.training.gradual_unfreeze is False


def test_external_test_configuration_is_parsed(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
data:
  max_attention_elements: 2000000
  external_tests:
    - name: ptmul-NR
      path: ptmul.pkl
      esm_cache_dir: esm
      esm_layout: paired_flat
      pad_node_features: true
      skip_missing_esm: true
      batch_size: 4
      max_attention_elements: 1000000
      ddg_multiplier: -1
      report_directions: true
training:
  external_test_interval: 5
  show_progress: false
  top_k_checkpoints: 5
""",
        encoding="utf-8",
    )

    config = load_config(path)
    external = config.data.external_tests[0]

    assert config.data.max_attention_elements == 2000000
    assert config.training.external_test_interval == 5
    assert config.training.show_progress is False
    assert config.training.top_k_checkpoints == 5
    assert external.name == "ptmul-NR"
    assert external.path == "ptmul.pkl"
    assert external.esm_layout == "paired_flat"
    assert external.pad_node_features is True
    assert external.skip_missing_esm is True
    assert external.batch_size == 4
    assert external.max_attention_elements == 1000000
    assert external.ddg_multiplier == -1
    assert external.report_directions is True


def test_training_esm_cache_fallbacks_are_parsed(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
data:
  esm_cache_dir: aligned_esm
  esm_cache_fallback_dirs:
    - recovered_esm
    - legacy_esm
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.data.esm_cache_dir == "aligned_esm"
    assert config.data.esm_cache_fallback_dirs == [
        "recovered_esm",
        "legacy_esm",
    ]


def test_train_ablation_sample_set_requires_a_path(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
data:
  sample_set: train_ablation_model
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="train_ablation_path is required"):
        load_config(path)


def test_train_ablation_sample_set_is_parsed(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
data:
  sample_set: train_ablation_model
  train_ablation_path: reference.pkl
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.data.sample_set == "train_ablation_model"
    assert config.data.train_ablation_path == "reference.pkl"


def test_pssm_sequence_feature_source_is_parsed(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
data:
  sequence_feature_source: pssm
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.data.sequence_feature_source == "pssm"
    assert config.model.sequence_dim == 41


def test_esm_pssm_projected_gated_configuration_is_parsed(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
data:
  sequence_feature_source: esm_pssm
model:
  sequence_dim: 1321
  sequence_fusion: projected_gated
  sequence_esm_dim: 1280
  sequence_pssm_dim: 41
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.data.sequence_feature_source == "esm_pssm"
    assert config.model.sequence_fusion == "projected_gated"
    assert config.model.sequence_dim == 1321


def test_projected_gated_dimensions_must_match(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model:
  sequence_dim: 1280
  sequence_fusion: projected_gated
  sequence_esm_dim: 1280
  sequence_pssm_dim: 41
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sequence_dim ="):
        load_config(path)


def test_esm_pssm_source_requires_projected_gated_fusion(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
data:
  sequence_feature_source: esm_pssm
model:
  sequence_dim: 1321
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be enabled together"):
        load_config(path)


def test_unknown_sequence_feature_source_is_rejected(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
data:
  sequence_feature_source: profile
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="sequence_feature_source must be one of",
    ):
        load_config(path)


def test_protein_split_level_is_parsed(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
data:
  split_level: protein
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.data.split_level == "protein"


def test_unknown_split_level_is_rejected(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
data:
  split_level: mutation
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="split_level must be one of"):
        load_config(path)


def test_top_k_checkpoints_must_be_positive(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
training:
  top_k_checkpoints: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="top_k_checkpoints must be positive"):
        load_config(path)


def test_flow_pretraining_configuration_is_parsed(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
pretraining:
  objective: flow
  latent_dim: 64
  central_weight: 0.5
  flow_weight: 2.0
  transport_weight: 0.25
  contrastive_weight: 0.3
  contrastive_temperature: 0.2
  local_crop_radius: 12.0
  local_crop_max_nodes: 128
  local_crop_neighbors: 32
model:
  freeze_structure_encoder: true
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.pretraining.objective == "flow"
    assert config.pretraining.latent_dim == 64
    assert config.pretraining.central_weight == 0.5
    assert config.pretraining.flow_weight == 2.0
    assert config.pretraining.transport_weight == 0.25
    assert config.pretraining.contrastive_weight == 0.3
    assert config.pretraining.contrastive_temperature == 0.2
    assert config.pretraining.local_crop_radius == 12.0
    assert config.pretraining.local_crop_max_nodes == 128
    assert config.pretraining.local_crop_neighbors == 32
    assert config.model.freeze_structure_encoder is True


def test_unknown_pretraining_objective_is_rejected(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
pretraining:
  objective: contrastive
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pretraining.objective"):
        load_config(path)
