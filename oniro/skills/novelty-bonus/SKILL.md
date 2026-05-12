---
name: novelty-bonus
description: Compute novelty score for a candidate descriptor via k-NN distance in archive descriptor space.
trigger: qd-archive-update
allowed_tools: [python]
output: json {novelty}
---

# novelty-bonus

Wraps `oniro.orchestrator.novelty.novelty(archive, descriptor, k=5)`.

## Use

Called from `qd-archive-update` to decide whether a gate-rejected mutation still
warrants a provisional slot. Threshold: `novelty > 0.7`.

## Output

```json
{ "novelty": 0.83 }
```
