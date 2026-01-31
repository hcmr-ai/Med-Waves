import torch


def adversarial_loss_G(D_fake: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    LSGAN generator loss: want D(fake) -> 1.

    Args:
        D_fake: Discriminator output on fake samples [B, 1, H', W']
        mask: Optional mask downsampled to match D_fake shape [B, 1, H', W']
    """
    if mask is not None:
        # Align shapes
        min_h = min(D_fake.shape[2], mask.shape[2])
        min_w = min(D_fake.shape[3], mask.shape[3])
        D_fake = D_fake[:, :, :min_h, :min_w]
        mask = mask[:, :, :min_h, :min_w]

        # Check if we have any valid pixels
        if not mask.any():
            return torch.tensor(0.0, device=D_fake.device)

        # Compute loss only on valid pixels
        loss_map = (D_fake - 1.0) ** 2
        return loss_map[mask].mean()
    else:
        return torch.mean((D_fake - 1.0) ** 2)


def adversarial_loss_D(D_real: torch.Tensor, D_fake: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    LSGAN discriminator loss:
      D(real) -> 1
      D(fake) -> 0

    Args:
        D_real: Discriminator output on real samples [B, 1, H', W']
        D_fake: Discriminator output on fake samples [B, 1, H', W']
        mask: Optional mask downsampled to match outputs [B, 1, H', W']
    """
    if mask is not None:
        # Align shapes
        min_h = min(D_real.shape[2], mask.shape[2])
        min_w = min(D_real.shape[3], mask.shape[3])
        D_real = D_real[:, :, :min_h, :min_w]
        D_fake = D_fake[:, :, :min_h, :min_w]
        mask = mask[:, :, :min_h, :min_w]

        # Check if we have any valid pixels
        if not mask.any():
            return torch.tensor(0.0, device=D_real.device)

        # Compute losses only on valid pixels
        loss_real_map = (D_real - 1.0) ** 2
        loss_fake_map = D_fake ** 2

        loss_real = loss_real_map[mask].mean()
        loss_fake = loss_fake_map[mask].mean()
    else:
        loss_real = torch.mean((D_real - 1.0) ** 2)
        loss_fake = torch.mean(D_fake ** 2)

    return 0.5 * (loss_real + loss_fake)
