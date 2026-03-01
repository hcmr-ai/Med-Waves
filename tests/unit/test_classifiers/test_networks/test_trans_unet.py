"""
Unit tests for TransUNetGeo and its building blocks.

Uses small channel/spatial dims so tests run quickly on CPU.
Run with: poetry run python -m pytest tests/unit/test_trans_unet.py -v
"""

import pytest
import torch

from src.classifiers.networks.trans_unet import (
    ConvBlock,
    DownBlock,
    DualUp,
    TransformerBranch,
    TransUNetGeo,
    UpBlock,
)


# ------------------------------------------------------------------ helpers

BATCH = 2
IN_CH = 4
BASE_CH = 16
BOTTLENECK = 64
PATCH_SIZE = 8
H, W = 64, 64


# ==================================================================
#  Building-block tests
# ==================================================================


class TestConvBlock:
    def test_output_shape(self):
        block = ConvBlock(IN_CH, BASE_CH)
        x = torch.randn(BATCH, IN_CH, H, W)
        out = block(x)
        assert out.shape == (BATCH, BASE_CH, H, W)

    def test_preserves_spatial_dims(self):
        block = ConvBlock(8, 16)
        x = torch.randn(1, 8, 33, 47)
        out = block(x)
        assert out.shape[-2:] == (33, 47)


class TestDownBlock:
    def test_output_and_skip_shapes(self):
        block = DownBlock(IN_CH, BASE_CH)
        x = torch.randn(BATCH, IN_CH, H, W)
        down, skip = block(x)

        assert skip.shape == (BATCH, BASE_CH, H, W), "Skip should keep original spatial size"
        assert down.shape == (BATCH, BASE_CH, H // 2, W // 2), "Down should halve spatial dims"

    def test_odd_spatial_dims(self):
        block = DownBlock(4, 8)
        x = torch.randn(1, 4, 33, 47)
        down, skip = block(x)
        assert skip.shape[-2:] == (33, 47)
        assert down.shape[-2:] == (17, 24)


class TestDualUp:
    def test_doubles_spatial_dims(self):
        up = DualUp(32, 16)
        x = torch.randn(BATCH, 32, 8, 8)
        out = up(x)
        assert out.shape == (BATCH, 16, 16, 16)


class TestUpBlock:
    def test_output_shape_with_skip(self):
        up = UpBlock(in_ch=32, skip_ch=16, out_ch=16)
        x = torch.randn(BATCH, 32, 8, 8)
        skip = torch.randn(BATCH, 16, 16, 16)
        out = up(x, skip)
        assert out.shape == (BATCH, 16, 16, 16)

    def test_handles_mismatched_skip(self):
        """UpBlock should interpolate when upsampled size doesn't match skip exactly."""
        up = UpBlock(in_ch=32, skip_ch=16, out_ch=16)
        x = torch.randn(BATCH, 32, 7, 7)
        skip = torch.randn(BATCH, 16, 15, 15)
        out = up(x, skip)
        assert out.shape == (BATCH, 16, 15, 15)


class TestTransformerBranch:
    def test_output_shape(self):
        branch = TransformerBranch(
            in_channels=IN_CH,
            emb_dim=BOTTLENECK,
            patch_size=PATCH_SIZE,
            num_layers=1,
            num_heads=4,
        )
        x = torch.randn(BATCH, IN_CH, H, W)
        out = branch(x)
        expected_h = H // PATCH_SIZE
        expected_w = W // PATCH_SIZE
        assert out.shape == (BATCH, BOTTLENECK, expected_h, expected_w)

    def test_variable_spatial_size(self):
        branch = TransformerBranch(
            in_channels=4,
            emb_dim=32,
            patch_size=8,
            num_layers=1,
            num_heads=4,
        )
        for h, w in [(64, 64), (48, 80)]:
            x = torch.randn(1, 4, h, w)
            out = branch(x)
            assert out.shape == (1, 32, h // 8, w // 8)


# ==================================================================
#  TransUNetGeo — single task
# ==================================================================


class TestTransUNetGeoSingleTask:
    @pytest.fixture()
    def model(self):
        return TransUNetGeo(
            in_channels=IN_CH,
            out_channels=1,
            auxiliary_tasks=["vhm0"],
            base_channels=BASE_CH,
            bottleneck_dim=BOTTLENECK,
            patch_size=PATCH_SIZE,
            num_layers=1,
        )

    def test_output_is_tensor(self, model):
        """Single-task mode should return a plain tensor, not a dict."""
        x = torch.randn(BATCH, IN_CH, H, W)
        out = model(x)
        assert isinstance(out, torch.Tensor)

    def test_output_shape(self, model):
        x = torch.randn(BATCH, IN_CH, H, W)
        out = model(x)
        assert out.shape == (BATCH, 1, H, W)

    def test_different_spatial_sizes(self, model):
        for h, w in [(64, 64), (48, 80)]:
            x = torch.randn(1, IN_CH, h, w)
            out = model(x)
            assert out.shape == (1, 1, h, w), f"Failed for input size ({h}, {w})"

    def test_gradients_flow(self, model):
        x = torch.randn(BATCH, IN_CH, H, W)
        out = model(x)
        loss = out.mean()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "Gradients should flow through the model"


# ==================================================================
#  TransUNetGeo — multi task
# ==================================================================


class TestTransUNetGeoMultiTask:
    @pytest.fixture()
    def model(self):
        return TransUNetGeo(
            in_channels=IN_CH,
            out_channels=1,
            auxiliary_tasks=["vhm0", "vtm02"],
            base_channels=BASE_CH,
            bottleneck_dim=BOTTLENECK,
            patch_size=PATCH_SIZE,
            num_layers=1,
        )

    def test_output_is_dict(self, model):
        x = torch.randn(BATCH, IN_CH, H, W)
        out = model(x)
        assert isinstance(out, dict)

    def test_output_keys(self, model):
        x = torch.randn(BATCH, IN_CH, H, W)
        out = model(x)
        assert set(out.keys()) == {"vhm0", "vtm02"}

    def test_each_head_shape(self, model):
        x = torch.randn(BATCH, IN_CH, H, W)
        out = model(x)
        for task_name, tensor in out.items():
            assert tensor.shape == (BATCH, 1, H, W), f"{task_name} shape mismatch"

    def test_heads_produce_different_outputs(self, model):
        x = torch.randn(BATCH, IN_CH, H, W)
        out = model(x)
        assert not torch.allclose(out["vhm0"], out["vtm02"]), (
            "Different heads should produce different outputs (untrained weights are random)"
        )

    def test_gradients_flow_all_heads(self, model):
        x = torch.randn(BATCH, IN_CH, H, W)
        out = model(x)
        loss = sum(t.mean() for t in out.values())
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() > 0 or "bias" in name


# ==================================================================
#  TransUNetGeo — MDN heads
# ==================================================================


class TestTransUNetGeoMDN:
    @pytest.fixture()
    def model(self):
        return TransUNetGeo(
            in_channels=IN_CH,
            out_channels=1,
            auxiliary_tasks=["vhm0", "vtm02"],
            base_channels=BASE_CH,
            bottleneck_dim=BOTTLENECK,
            patch_size=PATCH_SIZE,
            num_layers=1,
            use_mdn=True,
        )

    def test_mdn_returns_tuple_per_task(self, model):
        x = torch.randn(BATCH, IN_CH, H, W)
        out = model(x)
        for task_name, components in out.items():
            assert isinstance(components, tuple) and len(components) == 3, (
                f"MDN head for {task_name} should return (pi, mu, sigma)"
            )

    def test_mdn_shapes(self, model):
        K = 3  # default number of mixture components
        x = torch.randn(BATCH, IN_CH, H, W)
        out = model(x)
        for task_name, (pi, mu, sigma) in out.items():
            assert pi.shape == (BATCH, K, H, W), f"{task_name} pi shape"
            assert mu.shape == (BATCH, K, H, W), f"{task_name} mu shape"
            assert sigma.shape == (BATCH, K, H, W), f"{task_name} sigma shape"

    def test_mdn_pi_sums_to_one(self, model):
        x = torch.randn(BATCH, IN_CH, H, W)
        out = model(x)
        for task_name, (pi, _, _) in out.items():
            sums = pi.sum(dim=1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
                f"{task_name}: mixture weights should sum to 1"
            )

    def test_mdn_sigma_positive(self, model):
        x = torch.randn(BATCH, IN_CH, H, W)
        out = model(x)
        for task_name, (_, _, sigma) in out.items():
            assert (sigma > 0).all(), f"{task_name}: sigma must be strictly positive"
