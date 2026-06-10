"""
Fisher Routing Diversity Loss.

Maximises between-modality routing diversity while minimising within-modality
routing variance — a proper Fisher Linear Discriminant criterion applied to
expert routing distributions.

Fisher criterion:
  J = between_var / (within_var + eps)
  Maximise J → routing is maximally discriminative across modalities

where:
  between_var = mean_m ||p_m - p_bar||^2     (class scatter)
  within_var  = mean over all i ||router_probs[i] - p_{ds_id[i]}||^2  (within-class scatter)

This strictly improves over plain between_var maximisation (old affinity loss):
  - Prevents degenerate solutions where routing collapses to very small values
    (small between_var AND small within_var → ratio stays high)
  - Encourages same-modality samples to route consistently (within_var penalised)
"""
import torch
import torch.nn.functional as F


def modality_affinity_loss(
    router_logits: torch.Tensor,
    ds_ids:        torch.Tensor,
    num_datasets:  int,
    eps:           float = 1e-6,
) -> torch.Tensor:
    """
    Args:
        router_logits: [B, K] raw router logits
        ds_ids:        [B]    integer modality index (0 .. num_datasets-1)
        num_datasets:  number of distinct modalities
        eps:           numerical stability for division

    Returns:
        scalar loss (negative Fisher criterion — minimise to maximise discriminability)
    """
    router_probs = F.softmax(router_logits, dim=-1)   # [B, K]

    modality_means = []
    valid_masks    = []
    for m in range(num_datasets):
        mask = (ds_ids == m)
        if mask.sum() > 0:
            modality_means.append(router_probs[mask].mean(dim=0))   # [K]
            valid_masks.append(mask)

    if len(modality_means) < 2:
        return torch.tensor(0.0, device=router_logits.device)

    stacked     = torch.stack(modality_means)          # [M, K]
    global_mean = stacked.mean(dim=0)                  # [K]

    # Between-modality scatter (class scatter in Fisher criterion)
    between_var = ((stacked - global_mean) ** 2).mean()

    # Within-modality scatter: how much same-modality samples disagree in routing
    within_parts = []
    for mean_m, mask in zip(modality_means, valid_masks):
        diff = router_probs[mask] - mean_m.unsqueeze(0)   # [n_m, K]
        within_parts.append((diff ** 2).mean())
    within_var = torch.stack(within_parts).mean()

    # Fisher criterion: maximise between / within → minimise negative ratio
    return -(between_var / (within_var + eps))
