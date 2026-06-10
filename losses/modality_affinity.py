"""
Modality Routing Affinity Loss.

Maximises between-modality routing diversity: different medical imaging
modalities should produce clearly distinct routing distributions.

Intuition (Fisher-criterion analogy):
  - Compute mean routing distribution per modality: p_m in R^K
  - Compute global mean: p_bar = mean(p_m)
  - Between-modality variance: V = mean_m ||p_m - p_bar||^2
  - We MAXIMISE V → minimise -V

This is an information-theoretic proxy for I(routing ; modality):
high between-modality variance ↔ routing is informative about modality.

Unlike the previous spec_head (cross-entropy), this loss:
  1. Requires no additional learnable parameters on the model.
  2. Directly penalises routing homogeneity rather than training a classifier.
  3. Is differentiable through router_probs → router weights.
"""
import torch
import torch.nn.functional as F


def modality_affinity_loss(
    router_logits: torch.Tensor,
    ds_ids:        torch.Tensor,
    num_datasets:  int,
) -> torch.Tensor:
    """
    Args:
        router_logits: [B, K] raw router logits
        ds_ids:        [B]    integer modality index (0 .. num_datasets-1)
        num_datasets:  number of distinct modalities

    Returns:
        scalar loss (negative between-modality routing variance)
    """
    router_probs = F.softmax(router_logits, dim=-1)   # [B, K]

    modality_means = []
    for m in range(num_datasets):
        mask = (ds_ids == m)
        if mask.sum() > 0:
            modality_means.append(router_probs[mask].mean(dim=0))  # [K]

    if len(modality_means) < 2:
        return torch.tensor(0.0, device=router_logits.device)

    stacked     = torch.stack(modality_means)          # [M, K]
    global_mean = stacked.mean(dim=0)                  # [K]

    # Between-modality variance
    between_var = ((stacked - global_mean) ** 2).mean()

    return -between_var   # minimise to maximise routing diversity
