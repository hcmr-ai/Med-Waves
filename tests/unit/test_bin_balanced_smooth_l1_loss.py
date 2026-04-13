import pytest
import torch
import torch.nn.functional as F

from src.commons.loss_functions.l1_loss import (
    masked_atlantic_low_bin_balanced_smooth_l1,
    masked_bin_balanced_smooth_l1,
)
from src.commons.losses_factory import compute_loss


def test_bin_balanced_smooth_l1_averages_nonempty_bins_equally():
    y_true = torch.zeros(1, 1, 1, 4)
    y_pred = torch.tensor([[[[0.1, 0.1, 1.0, 1.0]]]])
    mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    vhm0 = torch.tensor([[[[0.5, 0.5, 4.0, 4.0]]]])

    loss = masked_bin_balanced_smooth_l1(
        y_pred,
        y_true,
        mask,
        vhm0,
        bin_thresholds=[1.0, 3.0],
        beta=0.3,
    )

    low_loss = F.smooth_l1_loss(
        y_pred[..., :2],
        y_true[..., :2],
        beta=0.3,
        reduction="none",
    ).mean()
    high_loss = F.smooth_l1_loss(
        y_pred[..., 2:],
        y_true[..., 2:],
        beta=0.3,
        reduction="none",
    ).mean()
    expected = (low_loss + high_loss) / 2.0

    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_bin_balanced_smooth_l1_ignores_empty_bins():
    y_true = torch.zeros(1, 1, 1, 2)
    y_pred = torch.tensor([[[[0.2, 0.4]]]])
    mask = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    vhm0 = torch.tensor([[[[0.5, 0.8]]]])

    loss = masked_bin_balanced_smooth_l1(
        y_pred,
        y_true,
        mask,
        vhm0,
        bin_thresholds=[1.0, 2.0, 3.0],
        beta=0.3,
    )
    expected = F.smooth_l1_loss(y_pred, y_true, beta=0.3, reduction="none").mean()

    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_atlantic_low_bin_balanced_smooth_l1_weights_low_bins_more():
    y_true = torch.zeros(1, 1, 1, 3)
    y_pred = torch.tensor([[[[0.2, 0.2, 1.0]]]])
    mask = torch.ones(1, 1, 1, 3, dtype=torch.bool)
    vhm0 = torch.tensor([[[[0.5, 1.5, 4.0]]]])

    weighted = masked_atlantic_low_bin_balanced_smooth_l1(
        y_pred, y_true, mask, vhm0, beta=0.3
    )
    equal = masked_bin_balanced_smooth_l1(y_pred, y_true, mask, vhm0, beta=0.3)

    assert weighted.item() < equal.item()


def test_loss_factory_supports_atlantic_low_bin_balanced_smooth_l1():
    y_true = torch.zeros(1, 1, 2, 2)
    y_pred = torch.ones(1, 1, 2, 2, requires_grad=True)
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    vhm0 = torch.tensor([[[[0.5, 1.5], [3.0, 6.0]]]])

    loss = compute_loss(
        "atlantic_low_bin_balanced_smooth_l1",
        y_pred,
        y_true,
        mask,
        vhm0_for_reconstruction=vhm0,
    )
    loss.backward()

    assert loss.dim() == 0
    assert y_pred.grad is not None
    assert y_pred.grad.abs().sum() > 0
