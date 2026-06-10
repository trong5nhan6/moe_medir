"""
Core MoE Specialization Module.

Architecture:
  Input [B, 1024]
      |
  Router (2-layer MLP) -> softmax -> top-k gating
      |
  Expert pool (num_experts × Expert MLP)
      |
  Weighted sum of top-k expert outputs
      |
  Output [B, expert_output_dim]

The router learns to assign different medical imaging modalities
to different experts, enabling modality-specific feature refinement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG


class Expert(nn.Module):
    """Single expert: Linear -> LayerNorm -> GELU -> Linear."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class MoESpecializationModule(nn.Module):
    """
    Mixture-of-Experts module with top-k sparse routing.

    Args:
        input_dim  : feature dimension from backbone (1024)
        num_experts: total number of experts (8)
        top_k      : how many experts are activated per token (2)
        hidden_dim : hidden dim inside each expert (512)
        output_dim : output dim of each expert (256)

    Forward returns:
        output       FloatTensor [B, output_dim]  — weighted expert outputs
        router_logits FloatTensor [B, num_experts] — raw logits before softmax
                                                     (used for load-balance loss)
    """
    def __init__(
        self,
        input_dim:  int = CFG.feature_dim,
        num_experts: int = CFG.num_experts,
        top_k:       int = CFG.top_k,
        hidden_dim:  int = CFG.expert_hidden,
        output_dim:  int = 256,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k

        # Router: 2-layer MLP -> num_experts logits
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts),
        )

        # Expert pool
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        """
        x: [B, input_dim]
        """
        B = x.size(0)

        # 1. Compute routing scores
        router_logits = self.router(x)                    # [B, num_experts]
        router_probs  = F.softmax(router_logits, dim=-1)  # [B, num_experts]

        # 2. Select top-k experts per sample
        topk_probs, topk_idx = router_probs.topk(
            self.top_k, dim=-1
        )                                                  # [B, top_k] each

        # Renormalise top-k weights to sum to 1
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # 3. Compute expert outputs for active (sample, expert) pairs
        # Flatten: process each expert once, gather relevant samples
        output = torch.zeros(B, self.experts[0].net[-1].out_features,
                             device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_ids = topk_idx[:, k]          # [B]
            weights    = topk_probs[:, k]        # [B]

            for e_idx in range(self.num_experts):
                mask = (expert_ids == e_idx)     # bool [B]
                if mask.any():
                    e_out = self.experts[e_idx](x[mask])   # [n, output_dim]
                    output[mask] += weights[mask].unsqueeze(-1) * e_out

        return output, router_logits
