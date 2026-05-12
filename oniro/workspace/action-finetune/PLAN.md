# action-finetune :: PLAN

## Goal
Action-conditioned next-slot IoU > 0.60 on held-out games, OOD next-slot MSE < 0.08.

## Order of work
1. Mount Open-X-Embodiment Apache subset.
2. Run Phase 2 finetune (250k steps).
3. Anneal data mix 90% OXE / 10% ARC → 60/40.

## Gates
- Action-conditioned IoU > 0.60.
- OOD next-slot MSE < 0.08.
- VLM caption recall@1 > 0.50.
