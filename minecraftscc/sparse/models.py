import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class DepthMDNHead(nn.Module):
    """
    Per-pixel Mixture Density Network head for depth.

    Input:
        features: [N, C, H, W]

    Output (per pixel):
        mixture_weights: [N, K, H, W]  (sum over K = 1)
        component_means: [N, K, H, W]  (positive, via exp => log-depth parameterization)
        component_scales: [N, K, H, W] (positive, via softplus + min_scale)
    """

    def __init__(self, input_channels: int, num_components: int = 3, min_scale: float = 1e-3):
        super().__init__()
        self.num_components = num_components
        self.min_scale = float(min_scale)

        # Light refinement of decoder features before prediction.
        self.refine = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=input_channels),
            nn.SiLU(),
        )

        # Predict [pi_logits, mean_log_depth, scale_logits] for each component => 3 * K channels.
        self.predict = nn.Conv2d(input_channels, 3 * num_components, kernel_size=1)

    def forward(self, features: torch.Tensor):
        refined = self.refine(features)
        params = self.predict(refined)  # [N, 3K, H, W]

        n, _, h, w = params.shape
        params = params.view(n, 3, self.num_components, h, w)

        pi_logits = params[:, 0]        # [N, K, H, W]
        mean_log  = params[:, 1]        # [N, K, H, W]
        scale_log = params[:, 2]        # [N, K, H, W]

        mixture_weights = F.softmax(pi_logits, dim=1)
        component_means = torch.exp(mean_log)  # log-depth -> positive depth
        component_scales = F.softplus(scale_log) + self.min_scale

        return mixture_weights, component_means, component_scales

def gaussian_mdn_nll_loss(
    mixture_weights: torch.Tensor,
    component_means: torch.Tensor,
    component_scales: torch.Tensor,
    depth_ground_truth: torch.Tensor,
    min_prob: float = 1e-10,
) -> torch.Tensor:
    """
    Negative log-likelihood for a per-pixel Gaussian mixture.

    Args:
        mixture_weights:  [N, K, H, W]  (softmax over K)
        component_means:  [N, K, H, W]
        component_scales: [N, K, H, W]  (sigma > 0)
        depth_ground_truth: [N, 1, H, W] or [N, H, W]
        min_prob: clamp to keep log(.) stable

    Returns:
        Scalar loss (mean over N,H,W).
    """
    depth = depth_ground_truth
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)

    # log N(d | mu_k, sigma_k^2) for each component k
    log_component_pdf = Normal(component_means, component_scales).log_prob(depth)  # [N, K, H, W]
    # log pi_k
    log_weights = torch.log(mixture_weights.clamp_min(min_prob))  # [N, K, H, W]
    # log Σ_k pi_k * N_k(d)  (stable)
    log_mixture_pdf = torch.logsumexp(log_weights + log_component_pdf, dim=1)  # [N, H, W]
    return (-log_mixture_pdf).mean()
