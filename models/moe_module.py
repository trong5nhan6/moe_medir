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
    → Gate weights are per-TOKEN softmax (normalize over selected experts per token),
      ensuring each token's total weight sums to 1 — scale-consistent with token_choice.
    → ~10% zero-coverage tokens (with capacity_factor=2.0) are handled by skip_proj.

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

        Gate normalization: per-TOKEN softmax over selected experts.
          gate[i, k] = softmax(router_logits[i, selected_experts_of_i])[k]
        → Each token's weights sum to 1, scale-consistent with token_choice.
        → Tokens with zero coverage get output=0 (skip_proj in MoEMedIR handles them).
        """
        B        = x.size(0)
        capacity = max(1, int(self.capacity_factor * B / self.num_experts))

        # Step 1: Each expert selects its top-c tokens → dispatch mask [B, K]
        dispatch = torch.zeros(B, self.num_experts,
                               dtype=torch.bool, device=x.device)
        for e_idx in range(self.num_experts):
            c = min(capacity, B)
            _, top_idx = router_logits[:, e_idx].topk(c, dim=0)  # [c]
            dispatch[top_idx, e_idx] = True

        # Step 2: Per-token gate weights
        # Mask out experts that did NOT select each token → softmax over selected only
        masked = router_logits.clone()                    # [B, K]
        masked[~dispatch] = float('-inf')
        gate = F.softmax(masked, dim=-1)                  # [B, K] per-token normalized

        # Zero out rows with no coverage (avoid nan from all-inf softmax)
        zero_coverage = ~dispatch.any(dim=1)              # [B]
        gate[zero_coverage] = 0.0

        # Step 3: Accumulate weighted expert outputs
        output = torch.zeros(B, self.output_dim, device=x.device, dtype=x.dtype)
        for e_idx in range(self.num_experts):
            mask = dispatch[:, e_idx]                     # [B] bool
            if not mask.any():
                continue
            e_out = self.experts[e_idx](x[mask])          # [n, D]
            output[mask] += gate[mask, e_idx].unsqueeze(-1) * e_out

        return output

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        router_logits = self.router(x)                           # [B, K]

        if self.routing_mode == "expert_choice":
            output = self._forward_expert_choice(x, router_logits)
        else:
            output = self._forward_token_choice(x, router_logits)

        return output, router_logits
