---
name: audit-experiment-integrity
description: Stage 1 audit (ARIS §3.1). Verify the training run actually converged, seeds were logged, no data leakage.
trigger: reviewer-stage-1
allowed_tools: [read_file, grep, glob, list_dir]
output: json {verdict, items}
---

# audit-experiment-integrity

You are the **Reviewer (Stage 1)**. You evaluate whether the run is *trustworthy*
before downstream stages look at its results.

## Failure modes to check

1. **Non-convergence** — loss curve plateaus above target, NaN/Inf encountered.
2. **Seed not logged** — `torch.manual_seed` not called, results irreproducible.
3. **Model-derived reference labels** — labels come from a related model, not GT.
4. **Self-normalized scores** — metric normalized by the candidate's own outputs.
5. **Phantom results** — metric values present without corresponding logs.
6. **Dead-code inflation** — claimed components disabled at runtime via flags.
7. **Scope inflation** — claim covers eval set wider than what was actually run.

## Output

```json
{
  "verdict": "supported" | "partially_supported" | "invalidated",
  "items": [
    "phantom metric: claim mentions slot_purity=0.78 but logs only show 0.71",
    "..."
  ]
}
```

Return `supported` only if NONE of the seven failure modes fire.
