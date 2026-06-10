import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG
from models.moe_module import MoESpecializationModule


class MoEMedIR(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_drop = nn.Dropout(0.1)
        self.moe = MoESpecializationModule(
            input_dim=CFG.feature_dim, num_experts=CFG.num_experts,
            top_k=CFG.top_k, hidden_dim=CFG.expert_hidden, output_dim=256)
        # Shared residual: ensures every token has a base representation
        # regardless of expert-choice coverage. Analogous to the "shared expert"
        # in DeepSeek-MoE — experts provide domain-specific delta on top of this.
        self.skip_proj = nn.Linear(CFG.feature_dim, 256, bias=False)
        # 2-layer projection head with non-linearity (matches MLP baseline capacity)
        self.proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, CFG.embed_dim),
            nn.LayerNorm(CFG.embed_dim),
        )
        # Modality classification head: predicts which dataset a sample comes from
        # based solely on routing distribution — trains router to be modality-discriminative
        self.spec_head = nn.Linear(CFG.num_experts, len(CFG.datasets))

    def forward(self, x):
        x        = self.input_drop(x)
        moe_out, router_logits = self.moe(x)
        combined = moe_out + self.skip_proj(x)       # residual fusion
        emb      = F.normalize(self.proj(combined), dim=-1)
        return emb, router_logits


class LinearBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(CFG.feature_dim, CFG.embed_dim)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1), None


class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CFG.feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, CFG.embed_dim),
            nn.LayerNorm(CFG.embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1), None
