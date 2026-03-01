"""
Unit tests for the hybrid MSE + Huber-tail loss.

Run with: poetry run python -m pytest tests/unit/test_mse_huber_tail_loss.py -v
"""

import torch
import pytest

from src.commons.loss_functions.huber_loss import (
    masked_mse_huber_tail_loss,
    _huber_per_pixel,
)


DEVICE = "cpu"
B, C, H, W = 2, 1, 16, 16


def _make_batch(vhm0_fill=2.0):
    """Create a synthetic batch with uniform VHM0."""
    y_pred = torch.randn(B, C, H, W)
    y_true = torch.randn(B, C, H, W)
    mask = torch.ones(B, C, H, W, dtype=torch.bool)
    vhm0 = torch.full((B, C, H, W), vhm0_fill)
    return y_pred, y_true, mask, vhm0


class TestHuberPerPixel:
    def test_small_errors_are_quadratic(self):
        error = torch.tensor([0.1, 0.2, 0.5])
        delta = 1.0
        result = _huber_per_pixel(error, delta)
        expected = 0.5 * (error**2) / delta
        assert torch.allclose(result, expected)

    def test_large_errors_are_linear(self):
        error = torch.tensor([2.0, 5.0, 10.0])
        delta = 1.0
        result = _huber_per_pixel(error, delta)
        expected = error - 0.5 * delta
        assert torch.allclose(result, expected)

    def test_continuity_at_delta(self):
        delta = 1.0
        error = torch.tensor([delta])
        from_quadratic = 0.5 * (error**2) / delta
        from_linear = error - 0.5 * delta
        assert torch.allclose(from_quadratic, from_linear, atol=1e-7)


class TestMseHuberTailLoss:
    def test_returns_scalar(self):
        loss = masked_mse_huber_tail_loss(*_make_batch())
        assert loss.dim() == 0

    def test_zero_loss_on_perfect_prediction(self):
        y = torch.randn(B, C, H, W)
        mask = torch.ones(B, C, H, W, dtype=torch.bool)
        vhm0 = torch.full((B, C, H, W), 5.0)
        loss = masked_mse_huber_tail_loss(y, y.clone(), mask, vhm0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_empty_mask_returns_zero(self):
        y_pred, y_true, _, vhm0 = _make_batch()
        mask = torch.zeros(B, C, H, W, dtype=torch.bool)
        loss = masked_mse_huber_tail_loss(y_pred, y_true, mask, vhm0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_all_bulk_equals_mse(self):
        """When all VHM0 < tail_threshold, loss should be pure MSE."""
        y_pred, y_true, mask, _ = _make_batch()
        vhm0 = torch.full((B, C, H, W), 2.0)  # well below default 8m threshold
        loss = masked_mse_huber_tail_loss(y_pred, y_true, mask, vhm0)

        expected_mse = ((y_pred - y_true) ** 2).mean()
        assert loss.item() == pytest.approx(expected_mse.item(), rel=1e-5)

    def test_all_tail_equals_weighted_huber(self):
        """When all VHM0 >= tail_threshold, loss should be tail_weight * Huber."""
        y_pred, y_true, mask, _ = _make_batch()
        vhm0 = torch.full((B, C, H, W), 12.0)
        tail_weight = 5.0
        delta = 0.5

        loss = masked_mse_huber_tail_loss(
            y_pred, y_true, mask, vhm0,
            tail_threshold=8.0, delta=delta, tail_weight=tail_weight,
        )

        error = torch.abs(y_pred - y_true)
        expected_huber = _huber_per_pixel(error, delta).mean()
        expected = tail_weight * expected_huber

        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_tail_weight_scales_tail_contribution(self):
        """Higher tail_weight should increase loss when tails are present."""
        y_pred, y_true, mask, _ = _make_batch()
        vhm0 = torch.full((B, C, H, W), 12.0)

        loss_w1 = masked_mse_huber_tail_loss(
            y_pred, y_true, mask, vhm0, tail_weight=1.0,
        )
        loss_w10 = masked_mse_huber_tail_loss(
            y_pred, y_true, mask, vhm0, tail_weight=10.0,
        )
        assert loss_w10.item() > loss_w1.item()

    def test_gradients_flow(self):
        y_pred = torch.randn(B, C, H, W, requires_grad=True)
        y_true = torch.randn(B, C, H, W)
        mask = torch.ones(B, C, H, W, dtype=torch.bool)
        vhm0 = torch.full((B, C, H, W), 10.0)

        loss = masked_mse_huber_tail_loss(y_pred, y_true, mask, vhm0)
        loss.backward()
        assert y_pred.grad is not None
        assert y_pred.grad.abs().sum() > 0

    def test_mixed_bulk_and_tail(self):
        """With mixed VHM0, loss should be between pure-bulk and pure-tail."""
        y_pred = torch.randn(B, C, H, W)
        y_true = torch.randn(B, C, H, W)
        mask = torch.ones(B, C, H, W, dtype=torch.bool)

        # Half pixels calm, half extreme
        vhm0 = torch.full((B, C, H, W), 3.0)
        vhm0[:, :, H // 2 :, :] = 12.0

        loss_mixed = masked_mse_huber_tail_loss(y_pred, y_true, mask, vhm0)

        vhm0_all_bulk = torch.full((B, C, H, W), 3.0)
        loss_bulk = masked_mse_huber_tail_loss(y_pred, y_true, mask, vhm0_all_bulk)

        vhm0_all_tail = torch.full((B, C, H, W), 12.0)
        loss_tail = masked_mse_huber_tail_loss(y_pred, y_true, mask, vhm0_all_tail)

        assert loss_mixed.item() != pytest.approx(loss_bulk.item(), rel=1e-3)
        assert loss_mixed.item() != pytest.approx(loss_tail.item(), rel=1e-3)

    def test_huber_tail_less_sensitive_to_large_errors(self):
        """
        For the same large prediction error on tail pixels,
        Huber should produce a smaller loss than MSE would.
        """
        y_true = torch.zeros(B, C, H, W)
        y_pred_large = torch.full((B, C, H, W), 5.0)  # large error
        mask = torch.ones(B, C, H, W, dtype=torch.bool)
        vhm0 = torch.full((B, C, H, W), 12.0)

        loss_hybrid = masked_mse_huber_tail_loss(
            y_pred_large, y_true, mask, vhm0,
            tail_weight=1.0, delta=1.0,
        )
        pure_mse = ((y_pred_large - y_true) ** 2).mean()

        assert loss_hybrid.item() < pure_mse.item(), (
            "Huber on large errors should be less than MSE"
        )
