import torch

from src.classifiers.lightning_trainer import WaveBiasCorrector
from src.classifiers.model_factory import create_model

_SMALL = dict(
    transunet_base_channels=4,
    transunet_bottleneck_dim=16,
    transunet_patch_size=4,
    transunet_num_layers=1,
    transunet_num_heads=4,
)


def test_moe_transunet_forward_shapes():
    model = create_model(
        "moe_transunet", in_channels=16, auxiliary_tasks=["vhm0"], num_experts=3, **_SMALL
    )
    x = torch.randn(2, 16, 64, 128)
    out = model(x)

    assert out["prediction"].shape == (2, 1, 64, 128)
    assert out["expert_outputs"].shape == (2, 3, 1, 64, 128)
    assert out["gate_weights"].shape == (2, 3, 64, 128)
    assert out["uncertainty"].shape == (2, 64, 128)
    assert out["expert_diversity_loss"] is None  # compute_diversity=False by default
    assert torch.allclose(
        out["gate_weights"].sum(dim=1), torch.ones(2, 64, 128), atol=1e-5
    )
    # Uncertainty in [0, 1]
    assert out["uncertainty"].min() >= 0.0
    assert out["uncertainty"].max() <= 1.0


def test_moe_transunet_multitask_shapes():
    model = create_model(
        "moe_transunet",
        in_channels=16,
        auxiliary_tasks=["vhm0", "vtm02"],
        num_experts=3,
        **_SMALL,
    )
    x = torch.randn(2, 16, 64, 128)
    out = model(x)

    assert isinstance(out["prediction"], dict)
    assert out["prediction"]["vhm0"].shape == (2, 1, 64, 128)
    assert out["prediction"]["vtm02"].shape == (2, 1, 64, 128)
    assert out["gate_weights"].shape == (2, 3, 64, 128)


def test_moe_transunet_physics_gate():
    model = create_model(
        "moe_transunet",
        in_channels=16,
        auxiliary_tasks=["vhm0"],
        num_experts=3,
        gate_input_mode="input_channels",
        gate_input_channels=[0, 1],
        **_SMALL,
    )
    x = torch.randn(2, 16, 64, 128)
    out = model(x)
    assert out["prediction"].shape == (2, 1, 64, 128)


def test_lightning_moe_auxiliary_losses_are_finite():
    module = WaveBiasCorrector(
        model_type="moe_transunet",
        in_channels=16,
        loss_type="mse",
        filters=[4, 8, 16],
        num_experts=3,
        gate_entropy_weight=0.01,
        gate_balance_weight=0.01,
        gate_prior_weight=0.01,
        expert_diversity_weight=0.01,
        gate_bin_edges=[1.0, 3.0],
        **_SMALL,
    )
    x = torch.randn(2, 16, 64, 128)
    y = torch.randn(2, 1, 64, 128)
    mask = torch.ones(2, 1, 64, 128, dtype=torch.bool)
    raw_vhm0 = torch.rand(2, 1, 64, 128) * 5.0

    out = module(x)
    pred, aux = module._extract_prediction_and_aux(out)
    supervised_loss, _ = module.compute_multi_task_loss(pred, y, mask, raw_vhm0)
    aux_loss = module._compute_gate_auxiliary_loss(aux, mask, raw_vhm0)

    assert torch.isfinite(supervised_loss)
    assert torch.isfinite(aux_loss)
