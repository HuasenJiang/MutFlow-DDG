from __future__ import annotations

import torch

from mutflow_ddg.config import ModelConfig
from mutflow_ddg.data.synthetic import make_synthetic_batch
from mutflow_ddg.models import MutFlowDDG


def main() -> None:
    for branch_mode in ("sequence", "structure", "both"):
        config = ModelConfig(
            branch_mode=branch_mode,
            node_dim=12,
            sequence_dim=16,
            edge_dim=6,
            hidden_dim=32,
            latent_dim=32,
            num_structure_layers=2,
            num_sequence_layers=1,
            num_attention_heads=4,
            flow_steps=4,
        )
        batch = make_synthetic_batch(
            node_dim=config.node_dim,
            sequence_dim=config.sequence_dim,
            edge_dim=config.edge_dim,
        )
        if branch_mode == "structure":
            batch.wild.sequence_features = None
            batch.mutant.sequence_features = None
        model = MutFlowDDG(config)
        output = model(batch)
        output["ddg"].sum().backward()
        print(
            "branch_mode",
            branch_mode,
            "ddg_shape",
            tuple(output["ddg"].shape),
            "trajectory_shape",
            tuple(output["trajectory"].shape),
        )
    print("smoke_test", "ok")


if __name__ == "__main__":
    main()
