from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mutflow_ddg.data.schema import GraphBatch

from .tensor_ops import (
    from_dense_batch,
    mutation_attention_pool,
    segment_mean,
    to_dense_batch,
)


class EdgeMessageLayer(nn.Module):
    """Small dependency-free message passing layer for the CoMDDG graph contract."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.node_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_message = (
            nn.Linear(edge_dim, hidden_dim, bias=False) if edge_dim > 0 else None
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        aggregate = torch.zeros_like(x)
        if edge_index.numel() > 0:
            source, target = edge_index
            message = self.node_message(x[source])
            if self.edge_message is not None:
                message = message + self.edge_message(edge_attr)
            aggregate.index_add_(0, target, message)
            degree = torch.bincount(target, minlength=x.shape[0]).clamp_min(1)
            aggregate = aggregate / degree.to(x.dtype).unsqueeze(-1)
        return self.norm(x + self.update(torch.cat([x, aggregate], dim=-1)))


class StructureEncoder(nn.Module):
    """Mutation-centered structural encoder inspired by Pythia's local task."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                EdgeMessageLayer(hidden_dim, edge_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.pool_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pool_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, graph: GraphBatch) -> dict[str, Tensor]:
        x = self.input_projection(graph.x)
        for layer in self.layers:
            x = layer(x, graph.edge_index, graph.edge_attr)
        center = x[graph.mutation_index]
        local = mutation_attention_pool(
            x,
            graph.batch,
            graph.mutation_index,
            self.pool_key,
            self.pool_query,
        )
        global_context = segment_mean(x, graph.batch, graph.num_graphs)
        graph_embedding = self.output(
            torch.cat([center, local, global_context], dim=-1)
        )
        return {"nodes": x, "center": center, "graph": graph_embedding}


class SequenceEncoder(nn.Module):
    """Encode per-residue representations for the sequence branch."""

    def __init__(
        self,
        sequence_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        fusion_mode: str = "linear",
        esm_dim: int = 1280,
        pssm_dim: int = 41,
    ):
        super().__init__()
        self.sequence_dim = sequence_dim
        self.fusion_mode = fusion_mode
        self.esm_dim = esm_dim
        self.pssm_dim = pssm_dim
        if fusion_mode == "linear":
            self.input_projection = nn.Linear(sequence_dim, hidden_dim)
            self.esm_projection = None
            self.pssm_projection = None
            self.modality_gate = None
            self.fusion_norm = None
        elif fusion_mode == "projected_gated":
            if sequence_dim != esm_dim + pssm_dim:
                raise ValueError(
                    "projected_gated fusion dimensions do not sum to "
                    "sequence_dim"
                )
            self.input_projection = None
            self.esm_projection = nn.Sequential(
                nn.LayerNorm(esm_dim),
                nn.Linear(esm_dim, hidden_dim),
                nn.GELU(),
            )
            self.pssm_projection = nn.Sequential(
                nn.LayerNorm(pssm_dim),
                nn.Linear(pssm_dim, hidden_dim),
                nn.GELU(),
            )
            self.modality_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            self.fusion_norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Unsupported sequence fusion mode: {fusion_mode}")
        self.local_conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=15, padding=7
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.pool_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pool_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, graph: GraphBatch) -> dict[str, Tensor]:
        if graph.sequence_features is None:
            raise ValueError(
                "SequenceEncoder requires per-residue representations in "
                "graph.sequence_features"
            )
        source = graph.sequence_features
        expected_dim = self.sequence_dim
        if source.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected {expected_dim}-dimensional sequence features, "
                f"got {source.shape[-1]}"
            )
        if self.fusion_mode == "linear":
            if self.input_projection is None:
                raise RuntimeError("Linear sequence projection is unavailable")
            x = self.input_projection(source)
        else:
            if (
                self.esm_projection is None
                or self.pssm_projection is None
                or self.modality_gate is None
                or self.fusion_norm is None
            ):
                raise RuntimeError("Projected gated fusion is unavailable")
            esm = self.esm_projection(source[:, : self.esm_dim])
            pssm = self.pssm_projection(source[:, self.esm_dim :])
            gate = self.modality_gate(torch.cat([esm, pssm], dim=-1))
            x = self.fusion_norm(gate * esm + (1.0 - gate) * pssm)
        dense, mask, local_index = to_dense_batch(x, graph.batch)
        local = self.local_conv(dense.transpose(1, 2)).transpose(1, 2)
        dense = self.transformer(
            dense + F.gelu(local), src_key_padding_mask=~mask
        )
        x = from_dense_batch(dense, mask)
        center = x[graph.mutation_index]
        attended = mutation_attention_pool(
            x,
            graph.batch,
            graph.mutation_index,
            self.pool_key,
            self.pool_query,
        )
        global_context = segment_mean(x, graph.batch, graph.num_graphs)
        graph_embedding = self.output(
            torch.cat([center, attended, global_context], dim=-1)
        )
        mutation_local_index = local_index[graph.mutation_index]
        return {
            "nodes": x,
            "center": center,
            "graph": graph_embedding,
            "mutation_local_index": mutation_local_index,
        }
