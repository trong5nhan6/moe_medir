"""
Expert Orthogonality Loss.

Penalises cosine similarity between every pair of expert weight matrices,
encouraging each expert to learn a distinct subspace of the feature space.

Reference: "Advancing Expert Specialization for Better MoE" (NeurIPS 2025).

Formula:
  L_orth = (1 / K(K-1)) * sum_{i != j} cos_sim(W_i, W_j)^2

where W_i is the flattened first-layer weight of expert i.
Minimising this → expert weight matrices become mutually orthogonal.
"""
import torch
import torch.nn.functional as F


def expert_orthogonality_loss(moe_module) -> torch.Tensor:
    """
    Args:
        moe_module: MoESpecializationModule instance

    Returns:
        scalar loss
    """
    # First Linear layer weights of each expert: [hidden_dim, input_dim]
    weights = torch.stack([
        e.net[0].weight for e in moe_module.experts
    ])                                                # [K, H, D]

    K      = weights.size(0)
    w_flat = F.normalize(weights.view(K, -1), dim=-1) # [K, D*H]

    # Gram matrix of pairwise cosine similarities: [K, K]
    gram     = w_flat @ w_flat.T
    off_diag = gram - torch.eye(K, device=gram.device)

    return off_diag.pow(2).sum() / (K * (K - 1))
