import torch

from mutflow_ddg.models.tensor_ops import from_dense_batch, to_dense_batch


def test_to_dense_batch_preserves_grouped_nodes_and_local_indices():
    x = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    batch = torch.tensor([0, 0, 1, 2, 2, 2])

    dense, mask, local_index = to_dense_batch(x, batch)

    assert dense.shape == (3, 3, 3)
    assert mask.tolist() == [
        [True, True, False],
        [True, False, False],
        [True, True, True],
    ]
    assert local_index.tolist() == [0, 1, 0, 0, 1, 2]
    assert torch.equal(from_dense_batch(dense, mask), x)


def test_to_dense_batch_backpropagates_to_sparse_nodes():
    x = torch.randn(5, 4, requires_grad=True)
    batch = torch.tensor([0, 0, 1, 1, 1])

    dense, mask, _ = to_dense_batch(x, batch)
    dense[mask].sum().backward()

    assert torch.equal(x.grad, torch.ones_like(x))
