import pytest
import torch

from scripts.pretrain_structure import AdaptiveTaskWeighting


def test_adaptive_task_weights_are_positive_bounded_and_scale_preserving():
    weighting = AdaptiveTaskWeighting(
        {
            "central_residue": 1.0,
            "flow": 1.0,
            "replacement": 0.5,
        }
    )
    with torch.no_grad():
        weighting.log_variances.copy_(
            torch.tensor([20.0, -20.0, -10.0])
        )

    weights = weighting.weights()
    total = weighting(
        {
            "central_residue": torch.tensor(2.5),
            "flow": torch.tensor(1e-5),
            "replacement": torch.tensor(1e-6),
        }
    )

    assert all(value >= 0 for value in weights.values())
    assert sum(weights.values()) == pytest.approx(2.5)
    assert max(weights.values()) <= 2.5
    assert torch.isfinite(total)
    assert total >= 0
