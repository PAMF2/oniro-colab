---
name: qd-archive-update
description: After a mutation passes (or just falls in an empty novelty cell), update the MAP-Elites archive.
trigger: orchestrator-after-gate
allowed_tools: [python]
output: json {placed, cell, tier, reason}
---

# qd-archive-update

Maintains the Quality-Diversity archive at `oniro/archive/`.

## Procedure

1. Compute the variant's descriptor:
   `(slot_purity, jepa_loss_bucket, action_acc)`.
2. Locate the MAP-Elites cell from the descriptor.
3. Call `QDArchive.insert(variant)`:
   - If gate `ACCEPT`: place as `canonical`, supersede the prior elite iff fitness
     strictly higher.
   - If gate `REJECT` but `novelty(descriptor) > 0.7` and the cell is empty:
     place as `provisional` (cannot supersede a parent, may seed mutations).
   - Otherwise: discard.
4. Append a line to `oniro/archive/lineage.jsonl` recording the decision.

## Output

```json
{
  "placed": true,
  "cell": [2, 1, 3],
  "tier": "canonical",
  "reason": "fitness 0.71 > prior elite 0.68"
}
```
