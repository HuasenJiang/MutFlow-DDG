import torch

from mutflow_ddg.config import LossConfig, ModelConfig
from mutflow_ddg.data.synthetic import make_synthetic_batch
from mutflow_ddg.losses import (
    MutFlowObjective,
    ddg_class_targets,
    structure_flow_pretraining_loss,
)
from mutflow_ddg.models import MutFlowDDG, StructureFlowPretrainingModel


def test_ddg_class_convention():
    ddg = torch.tensor([-1.0, 0.0, 1.0])
    target = ddg_class_targets(ddg, 0.5, "positive_destabilizing")
    assert target.tolist() == [0, 1, 2]


def test_joint_objective_is_finite():
    model_config = ModelConfig(
        node_dim=12,
        sequence_dim=16,
        edge_dim=6,
        hidden_dim=32,
        latent_dim=32,
        num_structure_layers=1,
        num_sequence_layers=1,
        num_attention_heads=4,
        flow_steps=2,
    )
    batch = make_synthetic_batch(
        node_dim=12,
        sequence_dim=16,
        edge_dim=6,
    )
    model = MutFlowDDG(model_config)
    direct = model(batch)
    reverse = model(batch.reversed())
    objective = MutFlowObjective(LossConfig())
    loss, parts = objective(direct, batch, reverse)
    assert torch.isfinite(loss)
    assert set(
        [
            "regression",
            "classification",
            "ranking",
            "symmetry",
            "transport",
            "uncertainty",
            "flow",
            "epistasis",
            "total",
        ]
    ).issubset(parts)
    loss.backward()


def test_structure_flow_pretraining_loss_is_finite():
    batch = make_synthetic_batch(node_dim=24, sequence_dim=16, edge_dim=6)
    model = StructureFlowPretrainingModel(
        node_dim=24,
        edge_dim=6,
        hidden_dim=32,
        latent_dim=32,
        num_layers=1,
        flow_steps=2,
    )
    output = model(batch.wild)
    loss, parts = structure_flow_pretraining_loss(output, batch.wild)

    assert torch.isfinite(loss)
    assert {
        "central_residue",
        "flow",
        "transport",
        "contrastive",
        "total",
    }.issubset(parts)
    loss.backward()
