"""TaskMicroLearner: small recursive net that compresses ARC demo pairs into a
task descriptor vector used to condition the main ONIRO dynamics.

Replaces the SHA1-hashed `action_disc` (which is arbitrary) with a *learned*
task descriptor that captures the rule shown across the demos. The micro net
is recursive: 4 internal cycles of (read demo slot pairs → mix → emit z).

Trained end-to-end with the main model. Also receives AlphaEvolve-Gödel
mutations *separately*, so it can be evolved independently of the backbone.

Approximate size at slot_dim=96, n_demos=3, n_cycles=4: ~1.4M params.
"""

from __future__ import annotations

import torch
from torch import nn


class TaskMicroLearner(nn.Module):
    def __init__(
        self,
        slot_dim: int = 96,
        K_slots: int = 8,
        n_demos: int = 3,
        hidden: int = 128,
        n_recursive_cycles: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        assert hidden % n_heads == 0
        self.slot_dim = slot_dim
        self.K = K_slots
        self.n_demos = n_demos
        self.hidden = hidden
        self.cycles = n_recursive_cycles
        self.n_heads = n_heads

        # Project each (slot_in, slot_out) pair into the micro working dim
        self.pair_proj = nn.Linear(2 * slot_dim, hidden)
        # Demo embeddings (slot 1..K, demo 1..N concatenated as a single sequence)
        self.pos_demo = nn.Parameter(torch.randn(1, n_demos * K_slots, hidden) * 0.02)

        # ONE shared transformer block — applied n_recursive_cycles times
        self.refine = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=2 * hidden,
            batch_first=True, activation="gelu", norm_first=True,
        )
        # Aggregator pool + project back to slot_dim
        self.aggregator = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, slot_dim),
        )
        # Optional gating: bias toward the SHA1 action or the learned z
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(
        self,
        demo_in_slots: torch.Tensor,
        demo_out_slots: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        demo_in_slots:  (B, N_demos, K, slot_dim)
        demo_out_slots: (B, N_demos, K, slot_dim)

        returns: dict with
            z_task: (B, slot_dim) — learned task descriptor
            cycles_traj: list of intermediate hidden states for introspection
        """
        if demo_in_slots.dim() != 4 or demo_out_slots.dim() != 4:
            raise ValueError("expected (B, N_demos, K, slot_dim)")
        B, N, K, D = demo_in_slots.shape
        if K != self.K or D != self.slot_dim:
            raise ValueError(f"shape mismatch K={K},D={D} vs cfg K={self.K},D={self.slot_dim}")

        pair = torch.cat([demo_in_slots, demo_out_slots], dim=-1)        # (B, N, K, 2D)
        pair = pair.reshape(B, N * K, 2 * D)                              # (B, N*K, 2D)
        h = self.pair_proj(pair)                                          # (B, N*K, hidden)
        h = h + self.pos_demo[:, : h.shape[1]]

        traj = [h]
        for _ in range(self.cycles):
            h = self.refine(h)                                            # shared recursive block
            traj.append(h)

        pooled = h.mean(dim=1)                                            # (B, hidden)
        z_task = self.aggregator(pooled)                                  # (B, slot_dim)
        return {"z_task": z_task, "cycles_traj": traj}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
