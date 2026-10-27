import torch

from mutflow_ddg.config import ModelConfig
from mutflow_ddg.data.synthetic import make_synthetic_batch
from mutflow_ddg.models import (
    MutFlowDDG,
    StructureFlowPretrainingModel,
    StructurePretrainingModel,
)


def small_config(branch_mode: str = "both") -> ModelConfig:
    return ModelConfig(
        branch_mode=branch_mode,
        node_dim=12,
        sequence_dim=16,
        edge_dim=6,
        hidden_dim=32,
        latent_dim=32,
        num_structure_layers=2,
        num_sequence_layers=1,
        num_attention_heads=4,
        flow_steps=3,
    )


def test_mutflow_forward_and_backward():
    config = small_config()
    batch = make_synthetic_batch(
        node_dim=config.node_dim,
        sequence_dim=config.sequence_dim,
        edge_dim=config.edge_dim,
    )
    model = MutFlowDDG(config)
    output = model(batch)
    assert output["ddg"].shape == (4,)
    assert output["trajectory"].shape == (4, config.flow_steps + 1, 32)
    assert output["class_logits"].shape == (4, 3)
    assert torch.isfinite(output["ddg"]).all()
    output["ddg"].sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mutflow_uses_full_sequence_graph_with_structure_patch():
    config = small_config()
    batch = make_synthetic_batch(
        nodes_per_graph=5,
        node_dim=config.node_dim,
        sequence_dim=config.sequence_dim,
        edge_dim=config.edge_dim,
    )
    full_sequence_batch = make_synthetic_batch(
        nodes_per_graph=9,
        node_dim=config.node_dim,
        sequence_dim=config.sequence_dim,
        edge_dim=config.edge_dim,
    )
    batch.wild.sequence_features = None
    batch.mutant.sequence_features = None
    batch.sequence_wild = full_sequence_batch.wild
    batch.sequence_mutant = full_sequence_batch.mutant

    model = MutFlowDDG(config)
    output = model(batch)
    output["ddg"].sum().backward()

    assert output["ddg"].shape == (4,)
    assert "sequence_wild" in output
    assert "structure_wild" in output
    assert torch.isfinite(output["ddg"]).all()


def test_sequence_only_forward_does_not_build_structure_branch():
    config = small_config(branch_mode="sequence")
    batch = make_synthetic_batch(
        node_dim=config.node_dim,
        sequence_dim=config.sequence_dim,
        edge_dim=config.edge_dim,
    )
    model = MutFlowDDG(config)
    output = model(batch)
    output["ddg"].sum().backward()

    assert model.sequence_encoder is not None
    assert model.structure_encoder is None
    assert model.state_fusion is None
    assert "sequence_wild" in output
    assert "structure_wild" not in output
    assert torch.isfinite(output["ddg"]).all()


def test_sequence_only_projected_gated_esm_pssm_fusion():
    config = ModelConfig(
        branch_mode="sequence",
        node_dim=12,
        sequence_dim=16,
        sequence_fusion="projected_gated",
        sequence_esm_dim=12,
        sequence_pssm_dim=4,
        edge_dim=6,
        hidden_dim=32,
        latent_dim=32,
        num_sequence_layers=1,
        num_attention_heads=4,
        flow_steps=3,
    )
    batch = make_synthetic_batch(
        node_dim=config.node_dim,
        sequence_dim=config.sequence_dim,
        edge_dim=config.edge_dim,
    )
    model = MutFlowDDG(config)
    output = model(batch)
    output["ddg"].sum().backward()

    encoder = model.sequence_encoder
    assert encoder is not None
    assert encoder.input_projection is None
    assert encoder.esm_projection is not None
    assert encoder.pssm_projection is not None
    assert encoder.modality_gate is not None
    assert any(
        parameter.grad is not None
        for parameter in encoder.modality_gate.parameters()
    )
    assert torch.isfinite(output["ddg"]).all()


def test_structure_only_forward_does_not_require_sequence_features():
    config = small_config(branch_mode="structure")
    batch = make_synthetic_batch(
        node_dim=config.node_dim,
        sequence_dim=config.sequence_dim,
        edge_dim=config.edge_dim,
    )
    batch.wild.sequence_features = None
    batch.mutant.sequence_features = None
    model = MutFlowDDG(config)
    output = model(batch)
    output["ddg"].sum().backward()

    assert model.sequence_encoder is None
    assert model.structure_encoder is not None
    assert model.state_fusion is None
    assert "sequence_wild" not in output
    assert "structure_wild" in output
    assert torch.isfinite(output["ddg"]).all()


def test_structure_pretraining_heads():
    config = ModelConfig(
        node_dim=24,
        sequence_dim=16,
        edge_dim=6,
        hidden_dim=32,
        latent_dim=32,
        num_structure_layers=2,
        num_sequence_layers=1,
        num_attention_heads=4,
        flow_steps=3,
    )
    batch = make_synthetic_batch(
        node_dim=config.node_dim,
        sequence_dim=config.sequence_dim,
        edge_dim=config.edge_dim,
    )
    model = StructurePretrainingModel(
        node_dim=config.node_dim,
        edge_dim=config.edge_dim,
        hidden_dim=config.hidden_dim,
        num_layers=2,
    )
    output = model(batch.wild)
    assert output["central_logits"].shape == (4, 20)


def test_structure_pretraining_uses_pythia_mask_token():
    batch = make_synthetic_batch(node_dim=28, sequence_dim=16, edge_dim=6)
    model = StructurePretrainingModel(
        node_dim=28,
        edge_dim=6,
        hidden_dim=32,
        num_layers=1,
    )

    corrupted = model.corrupt_graph(
        batch.wild,
        mask_ratio=1.0,
        random_replace_ratio=0.0,
        edge_noise_std=0.0,
    )
    center_tokens = corrupted.x[batch.wild.mutation_index, :22]

    assert torch.all(center_tokens[:, 21] == 1)
    assert torch.all(center_tokens[:, :21] == 0)


def test_structure_flow_pretraining_outputs():
    config = ModelConfig(
        node_dim=24,
        sequence_dim=16,
        edge_dim=6,
        hidden_dim=32,
        latent_dim=32,
        num_structure_layers=2,
        num_sequence_layers=1,
        num_attention_heads=4,
        flow_steps=3,
    )
    batch = make_synthetic_batch(
        node_dim=config.node_dim,
        sequence_dim=config.sequence_dim,
        edge_dim=config.edge_dim,
    )
    model = StructureFlowPretrainingModel(
        node_dim=config.node_dim,
        edge_dim=config.edge_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_layers=2,
        flow_steps=config.flow_steps,
    )
    output = model(batch.wild)

    assert output["central_logits"].shape == (4, 20)
    assert output["predicted_velocity"].shape == (4, config.latent_dim)
    assert output["target_velocity"].shape == (4, config.latent_dim)
    assert output["trajectory"].shape == (4, config.flow_steps + 1, 32)


def test_freezing_structure_encoder_blocks_structure_gradients():
    config = small_config()
    batch = make_synthetic_batch(
        node_dim=config.node_dim,
        sequence_dim=config.sequence_dim,
        edge_dim=config.edge_dim,
    )
    model = MutFlowDDG(config)
    model.freeze_structure_encoder()
    output = model(batch)
    output["ddg"].sum().backward()

    structure_parameters = list(model.structure_encoder.parameters())
    assert structure_parameters
    assert all(parameter.grad is None for parameter in structure_parameters)
    assert any(
        parameter.grad is not None
        for parameter in model.state_fusion.parameters()
    )


def test_reversed_batch_swaps_endpoints_and_sign():
    batch = make_synthetic_batch()
    full_sequence_batch = make_synthetic_batch(nodes_per_graph=18)
    batch.sequence_wild = full_sequence_batch.wild
    batch.sequence_mutant = full_sequence_batch.mutant
    reverse = batch.reversed()
    assert reverse.wild is batch.mutant
    assert reverse.mutant is batch.wild
    assert reverse.sequence_wild is batch.sequence_mutant
    assert reverse.sequence_mutant is batch.sequence_wild
    assert torch.equal(reverse.ddg, -batch.ddg)
