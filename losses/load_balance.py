"""
Load Balance Loss — Switch Transformer (Fedus et al. 2022).

Prevents expert collapse: ensures all experts are used roughly equally.

Formula:
  L_lb = num_experts * sum_i( f_i * p_i )
  f_i  = fraction of tokens dispatched to expert i  (from argmax)
  p_i  = mean router probability for expert i       (differentiable)

f_i * p_i is minimised when load is balanced (each f_i ≈ 1/K).
Multiplying by K keeps the loss scale invariant to num_experts.
"""
import torch
import torch.nn.functional as F


def load_balance_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Args:
        router_logits: [B, num_experts]  raw logits before softmax

    Returns:
        scalar loss
    """
    num_experts = router_logits.size(-1)
    probs = F.softmax(router_logits, dim=-1)          # [B, K]

    # f_i: fraction of tokens routed to expert i (hard, via argmax)
    top1_idx = router_logits.argmax(dim=-1)           # [B]
    one_hot  = F.one_hot(top1_idx, num_experts).float()  # [B, K]
    f = one_hot.mean(dim=0)                           # [K]

    # p_i: mean router probability per expert (soft, differentiable)
    p = probs.mean(dim=0)                             # [K]

    return num_experts * (f * p).sum()
