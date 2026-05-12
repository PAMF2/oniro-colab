"""Target-sparsity regularizer for adjacency matrices.

Plain L1 on σ(adj) drives ALL edges to 0 → sparse graph collapses to no-op.
Target-sparsity penalizes deviation from a target activation rate (e.g. 30%
edges active). The graph stays informative AND sparse.

penalty(adj) = λ · (mean(σ(adj)) - target)²
"""

from __future__ import annotations

import torch


def target_sparsity_loss(
    adj_logits: torch.Tensor,
    target: float = 0.3,
    lam: float = 1.0,
) -> torch.Tensor:
    """
    adj_logits: pre-sigmoid adjacency tensor (any shape).
    target: desired mean of σ(adj_logits), e.g. 0.3 = 30% active edges.
    lam: weight on the squared deviation.
    """
    activations = torch.sigmoid(adj_logits)
    mean_act = activations.mean()
    dev = mean_act - target
    return lam * dev * dev
