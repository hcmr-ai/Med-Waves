import math

import torch


def mdn_nll_loss(pi, mu, sigma, y, mask=None, eps=1e-6):
    """
    Numerically stable MDN NLL loss.
    pi, mu, sigma: [B, K, H, W]
    y:             [B, 1, H, W]
    mask:          [B, 1, H, W]  (1=ocean, 0=land), optional

    returns scalar NLL loss
    """
    # ===== CLEAN NaN VALUES FIRST (from land pixels) =====
    pi = torch.nan_to_num(pi, nan=1.0/pi.shape[1], posinf=1.0, neginf=0.0)  # Use uniform distribution for NaN
    mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = torch.nan_to_num(sigma, nan=1.0, posinf=10.0, neginf=eps)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Clamp sigma to prevent numerical issues
    sigma = torch.clamp(sigma, min=eps, max=100.0)

    # Expand y to match mixture dimension
    y_expanded = y.expand_as(mu)    # [B, K, H, W]

    # Compute log Gaussian density (more stable than exp then log)
    # log N(y|mu,sigma) = -0.5*log(2*pi) - log(sigma) - 0.5*((y-mu)/sigma)^2
    log_2pi = math.log(2 * math.pi)
    z_score = (y_expanded - mu) / sigma
    log_normal = -0.5 * log_2pi - torch.log(sigma) - 0.5 * z_score**2

    # Add log mixture weights: log(pi * N(y|mu,sigma)) = log(pi) + log(N)
    log_pi = torch.log(pi + eps)
    log_weighted = log_pi + log_normal  # [B, K, H, W]

    # Log-sum-exp trick for numerical stability
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    max_log_weighted = log_weighted.max(dim=1, keepdim=True)[0]
    log_sum_exp = torch.sum(torch.exp(log_weighted - max_log_weighted), dim=1)
    log_likelihood = max_log_weighted.squeeze(1) + torch.log(log_sum_exp + eps)  # [B, H, W]

    # Check for NaN before masking (debug)
    if torch.isnan(log_likelihood).any():
        print("WARNING: NaN in MDN log_likelihood BEFORE masking!")
        print(f"  pi range: [{pi.min():.4f}, {pi.max():.4f}], has_nan: {torch.isnan(pi).any()}")
        print(f"  mu range: [{mu.min():.4f}, {mu.max():.4f}], has_nan: {torch.isnan(mu).any()}")
        print(f"  sigma range: [{sigma.min():.4f}, {sigma.max():.4f}], has_nan: {torch.isnan(sigma).any()}")
        print(f"  log_sum_exp range: [{log_sum_exp.min():.4f}, {log_sum_exp.max():.4f}]")
        # Replace NaN with large negative value (high loss)
        log_likelihood = torch.nan_to_num(log_likelihood, nan=-10.0)

    # Apply mask
    if mask is not None:
        mask = mask.squeeze(1)
        valid_mask = mask.bool()
        if not valid_mask.any():
            return torch.tensor(0.0, device=pi.device, requires_grad=True)
        log_likelihood_masked = log_likelihood[valid_mask]
    else:
        log_likelihood_masked = log_likelihood

    # NLL: -mean(log p(y|x))
    # Clamp log_likelihood to prevent negative loss
    log_likelihood_masked = torch.clamp(log_likelihood_masked, max=0.0)
    nll = -log_likelihood_masked.mean()

    # Check for NaN in output
    if torch.isnan(nll):
        print("WARNING: NaN in MDN NLL output!")
        return torch.tensor(0.0, device=pi.device, requires_grad=True)

    return nll
