from oniro.losses.vicreg import vicreg
from oniro.losses.jepa import jepa_loss
from oniro.losses.sae_loss import sae_loss
from oniro.losses.vlm_ce import vlm_ce_loss
from oniro.losses.contrastive import info_nce, slot_infonce
from oniro.losses.grid_ce import grid_ce_loss
from oniro.losses.dis import dis_loss, make_dis_targets
from oniro.losses.target_sparsity import target_sparsity_loss

__all__ = [
    "vicreg", "jepa_loss", "sae_loss", "vlm_ce_loss",
    "info_nce", "slot_infonce", "grid_ce_loss",
    "dis_loss", "make_dis_targets",
    "target_sparsity_loss",
]
