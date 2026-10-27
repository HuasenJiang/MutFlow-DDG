from __future__ import annotations

import torch

from .schema import GraphBatch, MutationBatch


def _chain_graph(
    batch_size: int,
    nodes_per_graph: int,
    node_dim: int,
    sequence_dim: int,
    edge_dim: int,
) -> GraphBatch:
    total_nodes = batch_size * nodes_per_graph
    x = torch.randn(total_nodes, node_dim)
    sequence_features = torch.randn(total_nodes, sequence_dim)
    batch = torch.arange(batch_size).repeat_interleave(nodes_per_graph)
    mutation_index = (
        torch.arange(batch_size) * nodes_per_graph + nodes_per_graph // 2
    )

    edge_parts = []
    for graph_id in range(batch_size):
        start = graph_id * nodes_per_graph
        src = torch.arange(start, start + nodes_per_graph - 1)
        dst = src + 1
        edge_parts.append(torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]))
    edge_index = torch.cat(edge_parts, dim=1)
    edge_attr = torch.randn(edge_index.shape[1], edge_dim)
    return GraphBatch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        mutation_index=mutation_index,
        sequence_features=sequence_features,
        residue_type=torch.randint(0, 20, (total_nodes,)),
        confidence=torch.rand(total_nodes, 1),
    )


def make_synthetic_batch(
    batch_size: int = 4,
    nodes_per_graph: int = 12,
    node_dim: int = 41,
    sequence_dim: int = 41,
    edge_dim: int = 16,
    mutation_dim: int = 0,
) -> MutationBatch:
    wild = _chain_graph(
        batch_size, nodes_per_graph, node_dim, sequence_dim, edge_dim
    )
    mutant = _chain_graph(
        batch_size, nodes_per_graph, node_dim, sequence_dim, edge_dim
    )
    mutation_features = (
        torch.randn(batch_size, mutation_dim) if mutation_dim > 0 else None
    )
    return MutationBatch(
        wild=wild,
        mutant=mutant,
        ddg=torch.randn(batch_size),
        mutation_features=mutation_features,
        protein_index=torch.arange(batch_size) // 2,
        num_mutations=torch.ones(batch_size, dtype=torch.long),
    )
