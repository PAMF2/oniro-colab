# ONIRO research wiki

Persistent variant ledger. Each `variants/V-XXXX.md` describes one mutation lineage node,
with typed edges (`extends | contradicts | supersedes | invalidates | depends_on`).

`failures.jsonl` is append-only; the executor MUST consult it before proposing a mutation,
and explicitly justify why its proposal is not a member of any `closed_branches`.

## Current state

- Variants: 0
- Pareto front: empty
- σ_noise: not yet calibrated
- Open audits: 0
