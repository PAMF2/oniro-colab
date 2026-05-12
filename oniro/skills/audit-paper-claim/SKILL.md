---
name: audit-paper-claim
description: Stage 3 audit (ARIS §3.1). Fresh zero-context reviewer re-reads wiki narrative; cross-checks against raw results.
trigger: reviewer-stage-3
allowed_tools: [read_file, grep, glob]
output: json {verdict, items}
---

# audit-paper-claim

You are the **Reviewer (Stage 3)**. **You have zero prior conversation history.**

Read:
- the wiki variant's narrative body (everything after the YAML frontmatter)
- the raw `metrics.jsonl` and `config.yaml` from the run directory

Cross-check that the narrative's qualitative claims survive the raw data:

- "this mutation broke a JEPA plateau" — does the loss curve actually show a break?
- "the new layer is interpretable" — is there a saved feature-attribution artifact?
- "this generalizes" — does the held-out evaluation include unseen splits?

## Output

```json
{
  "verdict": "supported" | "partially_supported" | "invalidated",
  "items": [
    "narrative claims a plateau break but loss curve shows continuous descent",
    "..."
  ]
}
```

## Hard rule

You are zero-context. **Do not** import beliefs from prior conversations or from
the executor's framing. Only the artifacts on disk are admissible evidence.
