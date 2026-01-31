"""
Learning Rate Scheduler Factory

Centralizes scheduler creation logic for PyTorch optimizers.
Compatible with PyTorch Lightning's scheduler configuration format.
"""

import math

import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup


def create_scheduler(
    optimizer,
    scheduler_config: dict,
    total_steps: int = None,
    max_epochs: int = None,
):
    """
    Factory function to create learning rate schedulers.

    Args:
        optimizer: PyTorch optimizer instance
        scheduler_config: Scheduler configuration dict with keys:
            - type: Scheduler type (required)
            - Additional type-specific parameters
        total_steps: Total training steps (required for warmup schedulers)
        max_epochs: Max training epochs (optional, for logging)

    Returns:
        tuple: (scheduler_config_dict, scheduler_metadata_dict)
            - scheduler_config_dict: Dict compatible with PyTorch Lightning's lr_scheduler format
              Contains 'scheduler', optionally 'monitor', 'interval', 'frequency'
            - scheduler_metadata_dict: Dict with scheduler metadata for logging (may be empty)

    Supported Scheduler Types:
        - "none": No scheduler (returns empty dicts)
        - "ReduceLROnPlateau": Reduce LR on metric plateau
        - "CosineAnnealingLR": Cosine annealing schedule
        - "StepLR": Step-based LR decay
        - "ExponentialLR": Exponential LR decay
        - "CosineAnnealingWarmupRestarts": Cosine annealing with warmup
        - "LambdaLR": Custom lambda-based schedule with warmup and cosine decay

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> config = {"type": "CosineAnnealingLR", "T_max": 50, "eta_min": 1e-6}
        >>> scheduler_cfg, metadata = create_scheduler(optimizer, config)
        >>> # Use with PyTorch Lightning:
        >>> # return {"optimizer": optimizer, "lr_scheduler": scheduler_cfg}
    """
    def get_float(key, default):
        """Helper to safely extract float values from config"""
        val = scheduler_config.get(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    if not scheduler_config or scheduler_config.get("type", "none") == "none":
        return {}, {}

    scheduler_type = scheduler_config["type"]
    scheduler_metadata = {}

    # ========== ReduceLROnPlateau ==========
    if scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "min"),
            factor=get_float("factor", 0.5),
            patience=int(scheduler_config.get("patience", 5)),
            min_lr=get_float("min_lr", 1e-7),
        )
        return {
            "scheduler": scheduler,
            "monitor": scheduler_config.get("monitor", "val_loss"),
        }, scheduler_metadata

    # ========== CosineAnnealingLR ==========
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_config.get("T_max", 50)),
            eta_min=get_float("eta_min", 1e-6),
        )
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }, scheduler_metadata

    # ========== StepLR ==========
    elif scheduler_type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config.get("step_size", 10)),
            gamma=get_float("gamma", 0.1),
        )
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }, scheduler_metadata

    # ========== ExponentialLR ==========
    elif scheduler_type == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=get_float("gamma", 0.1)
        )
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }, scheduler_metadata

    # ========== CosineAnnealingWarmupRestarts ==========
    elif scheduler_type == "CosineAnnealingWarmupRestarts":
        if total_steps is None:
            raise ValueError(
                "total_steps is required for CosineAnnealingWarmupRestarts scheduler. "
                "Pass trainer.estimated_stepping_batches when calling create_scheduler()."
            )

        warmup_ratio = get_float("warmup_steps", 0.1)
        warmup_steps = int(warmup_ratio * total_steps)

        # Metadata for logging
        scheduler_metadata = {
            "total_steps": total_steps,
            "max_epochs": max_epochs,
            "warmup_ratio": warmup_ratio,
            "warmup_steps_calculated": warmup_steps
        }

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "scheduler": scheduler,
            "interval": "step",  # CRITICAL — warmup MUST be per-step
            "frequency": 1,
        }, scheduler_metadata

    # ========== LambdaLR (with warmup + cosine decay) ==========
    elif scheduler_type == "LambdaLR":
        if total_steps is None:
            raise ValueError(
                "total_steps is required for LambdaLR scheduler. "
                "Pass trainer.estimated_stepping_batches when calling create_scheduler()."
            )

        warmup_frac = get_float("warmup_steps", 0.1)
        warmup_steps = int(warmup_frac * total_steps)
        print(f"Warmup steps: {warmup_steps}, Total steps: {total_steps}")

        def lr_lambda(step):
            # Warmup: linear increase
            if step < warmup_steps:
                return max(0.01, step / max(1, warmup_steps))

            # Cosine decay: smooth decrease
            progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }, scheduler_metadata

    else:
        # Unknown scheduler type, return empty (no scheduler)
        print(f"Warning: Unknown scheduler type '{scheduler_type}', proceeding without scheduler")
        return {}, {}
