"""Statistics summarize tokens, including the conventional even-count median."""

import pytest
import torch

from experiments.train_infra.plot_training_logits import relative_error_statistics


def test_token_statistics_use_mean_and_interpolated_median():
    x = torch.tensor([[3., 4.], [0., 2.], [2., 0.], [0., 1.]])
    y = x * torch.tensor([1.1, 1.2, 1.4, 2.]).unsqueeze(-1)
    relative = (y - x).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-30)
    assert relative_error_statistics(relative) == pytest.approx({
        "token_relative_l2_max": 1., "token_relative_l2_mean": .425, "token_relative_l2_median": .3,
    })
    with pytest.raises(AssertionError):
        relative_error_statistics(torch.tensor([float("nan")]))
