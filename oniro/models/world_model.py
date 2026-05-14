"""ARC-AGI-3 world model: dynamics + reward + value (v41.1).

Action-conditioned next-state predictor + reward head + state-value head.
Trained jointly with the policy via:

    L = L_grid (Socrates) + L_dynamics_MSE + L_reward_MSE + L_value_TD

Plus curiosity bonus (ensemble disagreement) added to reward at run time.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DynamicsPredictor(nn.Module):
    """Predicts next-state latent given (urm_hidden, action_embed).

    A 2-layer transformer with cross-attention from action to state tokens,
    then MLP that mixes pooled-state + action → next-state-pool prediction.
    """

    def __init__(self, d_model: int = 896, n_heads: int = 8,
                 n_layers: int = 3, ffn_mult: int = 4):
        super().__init__()
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_mult * d_model,
            batch_first=True, activation='gelu', norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, state_tokens: torch.Tensor,
                action_token: torch.Tensor) -> torch.Tensor:
        """state_tokens: (B, T, D). action_token: (B, 1, D).
        Returns next_state_pool: (B, D).
        """
        seq = torch.cat([action_token, state_tokens], dim=1)
        out = self.trunk(seq)
        pooled = out.mean(dim=1)
        return self.proj(self.norm(pooled))


class ValueHead(nn.Module):
    """V(s) ∈ R. MLP from URM pooled state."""

    def __init__(self, d_model: int = 896, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state_pool: torch.Tensor) -> torch.Tensor:
        return self.net(state_pool).squeeze(-1)


class RewardHead(nn.Module):
    """r(s, a). MLP from concat(state_pool, action_token)."""

    def __init__(self, d_model: int = 896, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state_pool: torch.Tensor,
                action_token: torch.Tensor) -> torch.Tensor:
        act = action_token.squeeze(1) if action_token.dim() == 3 else action_token
        return self.net(torch.cat([state_pool, act], dim=-1)).squeeze(-1)


class PolicyHead(nn.Module):
    """π(a | s). Output logits over 6 discrete actions + click MLP.

    Returns dict {'discrete_logits': (B, 6), 'click_xy_mean': (B, 2)}.
    """

    def __init__(self, d_model: int = 896, hidden: int = 256,
                 n_discrete: int = 6):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_discrete),
        )
        self.click = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
            nn.Sigmoid(),       # clamp to [0, 1]
        )

    def forward(self, state_pool: torch.Tensor) -> dict:
        return {
            "discrete_logits": self.disc(state_pool),
            "click_xy_mean": self.click(state_pool),
        }


class CuriosityEnsemble(nn.Module):
    """K bootstrap predictors of next-state latent. Variance = intrinsic reward.

    Each predictor is a tiny MLP: (state_pool + action_embed) → next_state_pool.
    Train each on a bootstrap subset of transitions; variance of predictions
    at inference is the curiosity signal.
    """

    def __init__(self, d_model: int = 896, hidden: int = 256, k_predictors: int = 5):
        super().__init__()
        self.k = k_predictors
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model + d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
            )
            for _ in range(k_predictors)
        ])

    def predict_all(self, state_pool: torch.Tensor,
                    action_token: torch.Tensor) -> torch.Tensor:
        """Returns (K, B, d_model). Caller computes variance over K."""
        act = action_token.squeeze(1) if action_token.dim() == 3 else action_token
        cat = torch.cat([state_pool, act], dim=-1)
        return torch.stack([p(cat) for p in self.predictors], dim=0)

    def curiosity_bonus(self, state_pool: torch.Tensor,
                        action_token: torch.Tensor) -> torch.Tensor:
        """Per-sample ensemble variance, scalar (B,)."""
        preds = self.predict_all(state_pool, action_token)
        var = preds.var(dim=0).mean(dim=-1)
        return var
