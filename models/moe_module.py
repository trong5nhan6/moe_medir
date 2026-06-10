"""
Core MoE Specialization Module.

Supports two routing modes (controlled by CFG.routing_mode):

  token_choice  (Switch Transformer, 2021)
    Each token selects its top-k experts via softmax + topk.
    Requires load-balance auxiliary loss to prevent expert collapse.

  expert_choice  (Zhou et al., NeurIPS 2022)
    Each expert selects its top-c tokens.
    c = capacity_factor * B / num_experts  (default capacity_factor=2.0)
    → Perfect load balance BY CONSTRUCTION — no auxiliary loss needed.
    → Variable per-token coverage (some tokens may be picked by 0 or many experts).
    → A shared residual projection (skip_proj in MoEMedIR) handles zero-coverage tokens.

Router: 2-layer MLP with noisy logits during training (Noisy Top-K, GShard 2020).
  During training: logits += Normal(0, softplus(W_noise * x))
  During eval:    no noise added
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
    Mixture-of-Experts with switchable token-choice / expert-choice routing.

    Args:
        input_dim      : backbone feature dimension (1536)
        num_experts    : total number of experts (8)
        top_k          : experts per token — used in token_choice mode
        hidden_dim     : hidden dim inside each expert (512)
        output_dim     : output dim of each expert (256)
        routing_mode   : "token_choice" | "expert_choice"
        capacity_factor: expert_choice capacity = capacity_factor * B / K

    Forward returns:
        output        FloatTensor [B, output_dim]
        router_logits FloatTensor [B, num_experts]  — raw logits (pre-noise, pre-softmax)
    """
    def __init__(
        self,
        input_dim:       int   = CFG.feature_dim,
        num_experts:     int   = CFG.num_experts,
        top_k:           int   = CFG.top_k,
        hidden_dim:      int   = CFG.expert_hidden,
        output_dim:      int   = 256,
        routing_mode:    str   = CFG.routing_mode,
        capacity_factor: float = CFG.capacity_factor,
    ):
        super().__init__()
        self.num_experts     = num_experts
        self.top_k           = top_k
        self.routing_mode    = routing_mode
        self.capacity_factor = capacity_factor
        self.output_dim      = output_dim

        # Router: 2-layer MLP → num_experts logits
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts),
        )

        # Noisy top-k: learnable noise parameter (GShard / ST-MoE)
        self.w_noise = nn.Linear(input_dim, num_experts, bias=False)

        # Expert pool
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])

    # ── Noisy logits helper ───────────────────────────────────────────────
    def _noisy_logits(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Add input-dependent Gaussian noise during training (GShard)."""
        if not self.training:
            return logits
        noise_std = F.softplus(self.w_noise(x))        # [B, K] positive std
        noise     = torch.randn_like(logits) * noise_std
        return logits + noise

    # ── Token-choice routing (Switch Transformer) ─────────────────────────
    def _forward_token_choice(self, x: torch.Tensor, router_logits: torch.Tensor):
        B = x.size(0)
        noisy   = self._noisy_logits(x, router_logits)
        probs   = F.softmax(noisy, dim=-1)              # [B, K]

        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros(B, self.output_dim, device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            expert_ids = topk_idx[:, k]
            weights    = topk_probs[:, k]
            for e_idx in range(self.num_experts):
                mask = (expert_ids == e_idx)
                if mask.any():
                    e_out = self.experts[e_idx](x[mask])
                    output[mask] += weights[mask].unsqueeze(-1) * e_out

        return output

    # ── Expert-choice routing (Zhou et al., NeurIPS 2022) ─────────────────
    def _forward_expert_choice(self, x: torch.Tensor, router_logits: torch.Tensor):
        """
        Each expert selects its top-c tokens.
        c = max(1, int(capacity_factor * B / K))

        Gate weights are per-expert softmax (sum-to-1 per expert, not per token).
        Tokens selected by multiple experts accumulate contributions.
        Tokens selected by 0 experts contribute 0 (handled by skip_proj in MoEMedIR).
        """
        B        = x.size(0)
        capacity = max(1, int(self.capacity_factor * B / self.num_experts))

        # router_logits: [B, K] → transpose to [K, B] for per-expert view
        expert_scores = router_logits.T                          # [K, B]

        output = torch.zeros(B, self.output_dim, device=x.device, dtype=x.dtype)

        for e_idx in range(self.num_experts):
            scores = expert_scores[e_idx]                        # [B]
            c      = min(capacity, B)
            top_scores, top_indices = scores.topk(c, dim=0)     # [c]
            gate   = F.softmax(top_scores, dim=0)               # [c] sum-to-1

            e_out  = self.experts[e_idx](x[top_indices])        # [c, D]
            output.index_add_(0, top_indices,
                              gate.unsqueeze(-1) * e_out)

        return output

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        router_logits = self.router(x)                           # [B, K]

        if self.routing_mode == "expert_choice":
            output = self._forward_expert_choice(x, router_logits)
        else:
            output = self._forward_token_choice(x, router_logits)

        return output, router_logits
