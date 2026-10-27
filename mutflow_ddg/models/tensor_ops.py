from __future__ import annotations

import torch
from torch import Tensor


def graph_count(batch: Tensor) -> int:
    return int(batch.max().item()) + 1 if batch.numel() else 0


def segment_mean(x: Tensor, batch: Tensor, num_graphs: int | None = None) -> Tensor:
    num_graphs = graph_count(batch) if num_graphs is None else num_graphs
    output = x.new_zeros((num_graphs, x.shape[-1]))
    output.index_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=num_graphs).clamp_min(1)
    return output / counts.to(x.dtype).unsqueeze(-1)


def segment_softmax(scores: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
    weights = torch.empty_like(scores)
    for graph_id in range(num_graphs):
        mask = batch == graph_id
        weights[mask] = torch.softmax(scores[mask], dim=0)
    return weights


def mutation_attention_pool(
    nodes: Tensor,
    batch: Tensor,
    mutation_index: Tensor,
    key_projection,
    query_projection,
) -> Tensor:
    queries = query_projection(nodes[mutation_index])
    keys = key_projection(nodes)
    scores = (keys * queries[batch]).sum(dim=-1) / (nodes.shape[-1] ** 0.5)
    weights = segment_softmax(scores, batch, mutation_index.numel())
    pooled = nodes.new_zeros((mutation_index.numel(), nodes.shape[-1]))
    pooled.index_add_(0, batch, nodes * weights.unsqueeze(-1))
    return pooled


def to_dense_batch(x: Tensor, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Convert grouped sparse nodes to padded [B, L, D] tensors.

    Returns dense nodes, a valid-node mask, and each sparse node's local index.
    """

    num_graphs = graph_count(batch)
    counts = torch.bincount(batch, minlength=num_graphs)
    max_nodes = int(counts.max().item())
    dense = x.new_zeros((num_graphs, max_nodes, x.shape[-1]))
    mask = torch.zeros(
        (num_graphs, max_nodes), dtype=torch.bool, device=x.device
    )
    starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    local_index = torch.arange(batch.numel(), device=batch.device) - starts[batch]
    dense[batch, local_index] = x
    mask[batch, local_index] = True
    return dense, mask, local_index


def from_dense_batch(dense: Tensor, mask: Tensor) -> Tensor:
    return dense[mask]
