# arc3-online-adapt :: PLAN

## Goal
ARC-AGI-3 dev score ≥ 0.30 by month 8; ≥ 0.50 by month 10; ≥ 0.70 at Kaggle submission.

## Order of work
1. Build OODBuffer with 10 splits across ARC-3 games and OXE held-out.
2. Hook MPCPlanner to ARC-3 env via local backend.
3. Run online adaptation with Gödel gate; track accept/reject ratio (target 10-35%).
4. Cold-test Kaggle pipeline (no internet, $50 cap, ≤9h).

## Gates
- Gödel accept rate in [10%, 35%].
- σ_noise refreshed weekly via no-op control run.
- Kaggle dry-run wall-clock < 9h.
