import torch

from src.classifiers.lightning_trainer import WaveBiasCorrector
from src.classifiers.model_factory import create_model


def test_moe_transunet_forward_shapes():
    model = create_model(
        "moe_transunet",
        in_channels=16,
        auxiliary_tasks=["vhm0"],
        transunet_base_channels=4,
        transunet_bottleneck_dim=16,
        transunet_patch_size=4,
        transunet_num_layers=1,
        transunet_num_heads=4,
        num_experts=3,
    )
    x = torch.randn(2, 16, 64, 128)

    out = model(x)

    assert out["prediction"].shape == (2, 1, 64, 128)
    assert out["expert_outputs"].shape == (2, 3, 1, 64, 128)
    assert out["gate_weights"].shape == (2, 3, 64, 128)
    assert torch.allclose(
        out["gate_weights"].sum(dim=1),
        torch.ones(2, 64, 128),
        atol=1e-5,
    )


def test_lightning_moe_auxiliary_losses_are_finite():
    module = WaveBiasCorrector(
        model_type="moe_transunet",
        in_channels=16,
        loss_type="mse",
        filters=[4, 8, 16],
        transunet_base_channels=4,
        transunet_bottleneck_dim=16,
        transunet_patch_size=4,
        transunet_num_layers=1,
        transunet_num_heads=4,
        num_experts=3,
        gate_entropy_weight=0.01,
        gate_balance_weight=0.01,
        gate_prior_weight=0.01,
        gate_bin_edges=[1.0, 3.0],
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
