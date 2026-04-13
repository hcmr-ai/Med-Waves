"""
Model Factory for Wave Bias Correction Models

Centralizes model creation logic to improve code organization and reusability.
"""

from src.classifiers.networks.bunet import (
    BU_Net_Geo,
    BU_Net_Geo_Nick,
    BU_Net_Geo_Nick_Enhanced,
)
from src.classifiers.networks.swin_unet import SwinUNetAgnostic
from src.classifiers.networks.trans_unet import MoETransUNetGeo, SimpleMLPGeo, TransUNetGeo
from src.classifiers.networks.trans_unet_gan import WaveTransUNetGAN


def create_model(
    model_type: str,
    in_channels: int = 3,
    filters: list = None,
    dropout: float = 0.2,
    add_vhm0_residual: bool = False,
    vhm0_channel_index: int = 0,
    upsample_mode: str = "nearest",
    use_mdn: bool = False,
    auxiliary_tasks: list = None,  # ['vhm0', 'vtm02']
    transunet_base_channels: int = 32,
    transunet_bottleneck_dim: int = 512,
    transunet_patch_size: int = 8,
    transunet_num_layers: int = 4,
    transunet_num_heads: int = 8,
    transformer_use_coord_pos_enc: bool = True,
    transformer_sea_mask_channel_index: int | None = None,
    num_experts: int = 3,
    gate_temperature: float = 1.0,
    gate_input_mode: str = "features",
    expert_dropout: float = 0.0,
    return_gate_maps: bool = True,
):
    """
    Factory function to create wave bias correction models.

    Args:
        model_type: Type of model architecture. Options:
            - "geo": BU_Net_Geo
            - "nick": BU_Net_Geo_Nick (default)
            - "enhanced": BU_Net_Geo_Nick_Enhanced
            - "transunet": TransUNetGeo
            - "moe_transunet": MoETransUNetGeo
            - "mlp": SimpleMLPGeo (per-pixel MLP baseline)
            - "swinunet": SwinUNetAgnostic
            - "transunet_gan": WaveTransUNetGAN
        in_channels: Number of input channels
        filters: List of filter sizes for conv layers (used by bunet models)
        dropout: Dropout rate (used by bunet models)
        add_vhm0_residual: Whether to add VHM0 as residual connection
        vhm0_channel_index: Index of VHM0 channel in input
        upsample_mode: Upsampling mode for decoder (used by enhanced model)
        use_mdn: Whether to use Mixture Density Network output

    Returns:
        Initialized model instance

    Raises:
        ValueError: If unsupported model_type is provided

    Example:
        >>> model = create_model("transunet", in_channels=4, use_mdn=True)
        >>> model = create_model("enhanced", filters=[16, 32, 64, 128])
    """
    if filters is None:
        filters = [16, 32, 64, 128, 256]

    if model_type == "geo":
        return BU_Net_Geo(
            in_channels=in_channels,
            filters=filters,
            dropout=dropout,
            add_vhm0_residual=add_vhm0_residual,
            vhm0_channel_index=vhm0_channel_index,
        )

    elif model_type == "enhanced":
        return BU_Net_Geo_Nick_Enhanced(
            in_channels=in_channels,
            filters=filters,
            dropout=dropout,
            add_vhm0_residual=add_vhm0_residual,
            vhm0_channel_index=vhm0_channel_index,
            upsample_mode=upsample_mode,
            use_mdn=use_mdn,
            auxiliary_tasks=auxiliary_tasks or ["vhm0"],
        )

    elif model_type == "transunet":
        return TransUNetGeo(
            in_channels=in_channels,
            out_channels=1,
            auxiliary_tasks=auxiliary_tasks or ["vhm0"],
            base_channels=transunet_base_channels,
            bottleneck_dim=transunet_bottleneck_dim,
            patch_size=transunet_patch_size,
            num_layers=transunet_num_layers,
            num_heads=transunet_num_heads,
            use_mdn=use_mdn,
            transformer_use_coord_pos_enc=transformer_use_coord_pos_enc,
            transformer_sea_mask_channel_index=transformer_sea_mask_channel_index,
        )

    elif model_type == "moe_transunet":
        if use_mdn:
            raise ValueError("moe_transunet does not support use_mdn=True.")
        return MoETransUNetGeo(
            in_channels=in_channels,
            out_channels=1,
            auxiliary_tasks=auxiliary_tasks or ["vhm0"],
            base_channels=transunet_base_channels,
            bottleneck_dim=transunet_bottleneck_dim,
            patch_size=transunet_patch_size,
            num_layers=transunet_num_layers,
            num_heads=transunet_num_heads,
            num_experts=num_experts,
            gate_temperature=gate_temperature,
            gate_input_mode=gate_input_mode,
            vhm0_channel_index=vhm0_channel_index,
            expert_dropout=expert_dropout,
            return_gate_maps=return_gate_maps,
            transformer_use_coord_pos_enc=transformer_use_coord_pos_enc,
            transformer_sea_mask_channel_index=transformer_sea_mask_channel_index,
        )

    elif model_type == "mlp":
        return SimpleMLPGeo(
            in_channels=in_channels,
            out_channels=1,
            auxiliary_tasks=auxiliary_tasks or ["vhm0"],
            hidden_dim=128,
            num_layers=3,
            dropout=0.0,
            use_mdn=use_mdn,
        )

    elif model_type == "swinunet":
        return SwinUNetAgnostic(
            img_size=(64, 64),
            in_chans=in_channels,
            num_classes=1,
            embed_dim=64,
            depths=(2, 2, 2, 2),
            num_heads=(2, 4, 8, 8),
            window_size=4,
            mlp_ratio=4.0,
        )

    elif model_type == "transunet_gan":
        return WaveTransUNetGAN(
            in_channels=in_channels,
            out_channels=1,
            base_channels=64,
            bottleneck_dim=1024,
            patch_size=16,
            num_layers=8,
            use_mdn=use_mdn,
        )

    elif model_type == "nick":
        # Default: BU_Net_Geo_Nick
        return BU_Net_Geo_Nick(
            in_channels=in_channels,
            filters=filters,
            dropout=dropout,
            add_vhm0_residual=add_vhm0_residual,
            vhm0_channel_index=vhm0_channel_index,
            use_mdn=use_mdn,
            auxiliary_tasks=auxiliary_tasks or ["vhm0"],
        )

    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            f"Supported types: 'geo', 'nick', 'enhanced', 'transunet', 'moe_transunet', 'mlp', 'swinunet', 'transunet_gan'"
        )
