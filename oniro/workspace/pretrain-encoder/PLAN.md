# pretrain-encoder :: PLAN

## Goal
Drive SigLIP+Slot encoder to slot purity > 0.70 on MOVi-E and SAE L0 in [25, 40].

## Order of work
1. Distill SigLIP-2 ViT-L → ViT-B (feature-matching, 2 days).
2. Mount Ego4D + HowTo100M shards.
3. Run Phase 1 pretrain (600k steps).
4. Run probe: slot purity + SAE L0 + reconstruction MSE on MOVi-E.

## Gates
- Slot purity > 0.70.
- SAE L0 in [25, 40], dead features < 5%.
- VICReg variance > 0.5 (no collapse).
