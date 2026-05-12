"""Top-level ONIRO composition.

Wires SigLIP encoder → Slot Attention → {EMA-SAE | Dynamics | VLM head | Curiosity}.
Forward returns a dict of all intermediate tensors needed by the loss aggregator.

Two presets:
    - "1b":   full architecture (target ~1.05B params)
    - "tiny": Colab proof-of-concept (~50M params, runs on a single T4)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from oniro.models.siglip import SigLIPEncoder
from oniro.models.slot_attention import SlotAttention
from oniro.models.sae import TopKSAE
from oniro.models.dynamics_mamba import DynamicsCore
from oniro.models.vlm_head import VLMHead
from oniro.models.curiosity import CuriosityEnsemble
from oniro.models.memory import SparseMemoryAttention
from oniro.models.grid_decoder import GridDecoder
from oniro.models.sparse_graph import SparseCausalSlotGraph
from oniro.models.micro_learner import TaskMicroLearner
from oniro.models.two_speed import TwoSpeedRecurrence
from oniro.models.identity_bias_decoder import IdentityBiasDecoder


@dataclass
class OniroConfig:
    image_size: int = 256
    patch_size: int = 16
    encoder_d: int = 768
    K_slots: int = 6
    slot_dim: int = 128
    sae_dict: int = 4096
    sae_topk: int = 32
    sae_ema_tau: float = 0.99
    dyn_d: int = 1024
    dyn_blocks: int = 24
    dyn_cross_at: tuple[int, ...] = (12, 23)
    n_actions: int = 5
    vlm_vocab: int = 32000
    vlm_d: int = 768
    vlm_layers: int = 12
    cur_K: int = 5
    cur_hidden: int = 512
    mem_size: int = 64
    mem_topk: int = 4
    mem_alpha: float = 0.2
    mem_heads: int = 4
    grid_decoder_size: int = 32
    grid_decoder_colors: int = 10
    grid_decoder_feat: int = 128
    use_identity_bias_decoder: bool = False
    sparse_graph_topk: int = 3
    sparse_graph_heads: int = 4
    sparse_graph_l1: float = 0.01
    n_recursive_cycles: int = 1
    adaptive_recursive_eps: float = 0.0
    use_micro_learner: bool = False
    micro_n_demos: int = 3
    micro_hidden: int = 128
    micro_cycles: int = 4
    micro_heads: int = 4
    use_two_speed: bool = False
    two_speed_h_blocks: int = 2
    two_speed_l_blocks: int = 2
    two_speed_h_period: int = 2
    two_speed_n_heads: int = 4
    use_mamba: bool = True
    tiny_encoder: bool = False

    @classmethod
    def tiny(cls) -> "OniroConfig":
        return cls(
            image_size=64, patch_size=8, encoder_d=192,
            K_slots=4, slot_dim=64,
            sae_dict=512, sae_topk=8, sae_ema_tau=0.99,
            dyn_d=192, dyn_blocks=4, dyn_cross_at=(2, 3),
            vlm_vocab=4096, vlm_d=192, vlm_layers=2,
            cur_K=3, cur_hidden=128,
            mem_size=16, mem_topk=2, mem_alpha=0.2, mem_heads=2,
            use_mamba=False, tiny_encoder=True,
        )


class Oniro(nn.Module):
    def __init__(self, cfg: OniroConfig | None = None):
        super().__init__()
        cfg = cfg or OniroConfig()
        self.cfg = cfg

        self.encoder = SigLIPEncoder(
            tiny=cfg.tiny_encoder, image_size=cfg.image_size,
            patch_size=cfg.patch_size, d_model=cfg.encoder_d,
        )
        self.slots = SlotAttention(
            num_slots=cfg.K_slots, dim=cfg.slot_dim, iters=3,
            input_dim=self.encoder.d_model,
        )
        self.sae = TopKSAE(d_in=cfg.slot_dim, dict_size=cfg.sae_dict, topk=cfg.sae_topk)

        self.action_disc = nn.Embedding(cfg.n_actions + 1, cfg.dyn_d)
        self.action_click = nn.Linear(2, cfg.dyn_d)

        self.dynamics = DynamicsCore(
            d_model=cfg.dyn_d, n_blocks=cfg.dyn_blocks, slot_dim=cfg.slot_dim,
            action_dim=cfg.dyn_d, cross_attn_at=cfg.dyn_cross_at, use_mamba=cfg.use_mamba,
        )
        # n_heads=8 works for all common d_model values divisible by 8 (192, 256, 384, 512, 768)
        self.vlm = VLMHead(
            vocab_size=cfg.vlm_vocab, slot_dim=cfg.slot_dim, d_model=cfg.vlm_d,
            n_layers=cfg.vlm_layers, n_heads=8,
        )
        self.curiosity = CuriosityEnsemble(
            K=cfg.cur_K, slot_dim=cfg.slot_dim, action_dim=cfg.dyn_d, hidden=cfg.cur_hidden,
        )
        self.memory = SparseMemoryAttention(
            slot_dim=cfg.slot_dim, memory_size=cfg.mem_size,
            topk=cfg.mem_topk, n_heads=cfg.mem_heads,
        )
        self.mem_alpha = cfg.mem_alpha
        if getattr(cfg, "use_identity_bias_decoder", False):
            self.grid_decoder = IdentityBiasDecoder(
                slot_dim=cfg.slot_dim, grid_size=cfg.grid_decoder_size,
                n_colors=cfg.grid_decoder_colors, feat_dim=cfg.grid_decoder_feat,
            )
            self._identity_decoder = True
        else:
            self.grid_decoder = GridDecoder(
                slot_dim=cfg.slot_dim, grid_size=cfg.grid_decoder_size,
                n_colors=cfg.grid_decoder_colors, feat_dim=cfg.grid_decoder_feat,
            )
            self._identity_decoder = False
        self.sparse_graph = SparseCausalSlotGraph(
            K=cfg.K_slots, slot_dim=cfg.slot_dim,
            topk=cfg.sparse_graph_topk, n_heads=cfg.sparse_graph_heads,
            l1_lambda=cfg.sparse_graph_l1,
        )
        if cfg.use_two_speed:
            self.two_speed = TwoSpeedRecurrence(
                slot_dim=cfg.slot_dim, action_dim=cfg.dyn_d,
                h_blocks=cfg.two_speed_h_blocks, l_blocks=cfg.two_speed_l_blocks,
                n_heads=cfg.two_speed_n_heads, h_period=cfg.two_speed_h_period,
            )
        else:
            self.two_speed = None
        if cfg.use_micro_learner:
            self.micro_learner = TaskMicroLearner(
                slot_dim=cfg.slot_dim, K_slots=cfg.K_slots,
                n_demos=cfg.micro_n_demos, hidden=cfg.micro_hidden,
                n_recursive_cycles=cfg.micro_cycles, n_heads=cfg.micro_heads,
            )
            # Inject z_task into action embedding via additive gating
            self.micro_to_action = nn.Linear(cfg.slot_dim, cfg.dyn_d)
        else:
            self.micro_learner = None

        self.register_buffer("ema_slots_w", torch.tensor(cfg.sae_ema_tau))
        self._ema_slot_buf: torch.Tensor | None = None

    def encode_slots(self, image: torch.Tensor) -> torch.Tensor:
        patches = self.encoder(image)
        return self.slots(patches)

    def embed_action(
        self, discrete: torch.Tensor | None = None, click: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if discrete is not None:
            a = self.action_disc(discrete)
            if click is not None:
                a = a + self.action_click(click)
            return a
        if click is not None:
            return self.action_click(click)
        raise ValueError("must pass at least one of discrete or click")

    def ema_update_slots(self, slots: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tau = float(self.ema_slots_w)
            cur = slots.detach()
            if self._ema_slot_buf is None or self._ema_slot_buf.shape != cur.shape:
                self._ema_slot_buf = cur.clone()
            else:
                self._ema_slot_buf.mul_(tau).add_(cur, alpha=1 - tau)
            return self._ema_slot_buf

    def forward(
        self,
        image: torch.Tensor,
        next_image: torch.Tensor | None = None,
        action_disc: torch.Tensor | None = None,
        action_click: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        demo_in_slots: torch.Tensor | None = None,
        demo_out_slots: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        slots_raw = self.encode_slots(image)
        # Sparse causal routing: slots interact via TopK sparse adjacency
        slots = slots_raw + self.sparse_graph(slots_raw)
        out: dict[str, torch.Tensor] = {"slots": slots, "slots_raw": slots_raw}
        out["sparse_graph_l1"] = self.sparse_graph.l1_penalty()
        out["sparse_adjacency"] = self.sparse_graph.adjacency().detach()

        ema = self.ema_update_slots(slots)
        recon, sparse_f = self.sae(ema)
        out["sae_recon"] = recon
        out["sae_features"] = sparse_f
        out["ema_slots"] = ema
        out["grid_logits_recon"] = self.grid_decoder(slots)

        if action_disc is not None or action_click is not None:
            a = self.embed_action(action_disc, action_click)

            # MicroLearner injection: compress demo pairs into z_task, add to action emb.
            if (self.micro_learner is not None
                    and demo_in_slots is not None and demo_out_slots is not None):
                micro_out = self.micro_learner(demo_in_slots, demo_out_slots)
                z_task = micro_out["z_task"]                   # (B, slot_dim)
                a_micro = self.micro_to_action(z_task)         # (B, dyn_d)
                gate = self.micro_learner.gate
                a = a + gate * a_micro
                out["z_task"] = z_task
                out["micro_gate"] = gate.detach()

            n_cycles = max(1, getattr(self.cfg, "n_recursive_cycles", 1))
            adapt_eps = getattr(self.cfg, "adaptive_recursive_eps", 0.0)

            # Two-speed (HRM-style) replaces plain recursive loop if enabled.
            if self.two_speed is not None:
                ts_out = self.two_speed(slots, a, n_cycles=n_cycles)
                trajectory = ts_out["trajectory"]
                cycles_run = n_cycles
                cur_slots = ts_out["slots_final"]
                pred_next_dyn = cur_slots
                mem_residual = self.memory(slots)
                out["pred_next_slots"] = cur_slots + self.mem_alpha * mem_residual
                out["pred_next_dyn"] = pred_next_dyn
                out["mem_residual"] = mem_residual
                out["action_emb"] = a
                out["n_cycles_run"] = torch.tensor(cycles_run)
                out["slot_trajectory"] = trajectory
                out["grid_logits_per_cycle"] = [self.grid_decoder(s) for s in trajectory[1:]]
                out["grid_logits_pred"] = out["grid_logits_per_cycle"][-1]
                out["_slots_for_memory_update"] = slots.detach()
                out["H_final"] = ts_out["H_final"]
                cur_pred = self.curiosity.predict_all(slots, a)
                out["curiosity_preds"] = cur_pred
                if next_image is not None:
                    with torch.no_grad():
                        target = self.encode_slots(next_image)
                    out["target_next_slots"] = target
                if text_tokens is not None:
                    out["vlm_logits"] = self.vlm(slots, text_tokens[:, :-1])
                    out["vlm_targets"] = text_tokens[:, 1:]
                return out

            cur_slots = slots
            trajectory = [cur_slots]
            cycles_run = 0
            for cycle in range(n_cycles):
                pred_next_dyn = self.dynamics(cur_slots, a)
                mem_residual = self.memory(cur_slots)
                refined = pred_next_dyn + self.mem_alpha * mem_residual
                if n_cycles > 1 and cycle < n_cycles - 1:
                    refined = refined + self.sparse_graph(refined)
                # Adaptive early exit
                if adapt_eps > 0.0 and cycle > 0:
                    delta = (refined - cur_slots).norm(dim=-1).mean().item()
                    cur_slots = refined
                    trajectory.append(cur_slots)
                    cycles_run = cycle + 1
                    if delta < adapt_eps:
                        break
                else:
                    cur_slots = refined
                    trajectory.append(cur_slots)
                    cycles_run = cycle + 1
            pred_next = cur_slots
            out["pred_next_slots"] = pred_next
            out["pred_next_dyn"] = pred_next_dyn
            out["mem_residual"] = mem_residual
            out["action_emb"] = a
            out["n_cycles_run"] = torch.tensor(cycles_run)
            out["slot_trajectory"] = trajectory
            # Deep Improvement Supervision: decode at each cycle for per-step CE
            out["grid_logits_per_cycle"] = [
                self.grid_decoder(s) for s in trajectory[1:]   # skip cycle 0 (input slots)
            ]

            out["_slots_for_memory_update"] = slots.detach()
            out["grid_logits_pred"] = self.grid_decoder(pred_next)

            cur_pred = self.curiosity.predict_all(slots, a)
            out["curiosity_preds"] = cur_pred

        if next_image is not None:
            with torch.no_grad():
                target = self.encode_slots(next_image)
            out["target_next_slots"] = target

        if text_tokens is not None:
            out["vlm_logits"] = self.vlm(slots, text_tokens[:, :-1])
            out["vlm_targets"] = text_tokens[:, 1:]

        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
