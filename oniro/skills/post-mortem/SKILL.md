---
name: post-mortem
description: On REJECT verdict, write a structured entry to the failure ledger so future executors skip the closed branch.
trigger: orchestrator-after-gate
allowed_tools: [python, bash]
output: jsonl-line appended to oniro/wiki/failures.jsonl
---

# post-mortem

When the Gödel gate REJECTS a mutation OR a downstream review invalidates a claim,
this skill records the cause so future Executor calls skip the closed branch.

## Output (single line, appended)

```json
{"id":"F-0142","variant":"V-0427","cause":"latent_collapse",
 "signature":{"vicreg_var":0.003,"slot_purity":0.31},
 "closed_branches":["sparse-prior-α>1e-3"],"date":"2026-08-14"}
```

## Field meanings

- `cause` — short slug: `latent_collapse | gate_borderline | nan_inf | regression`...
- `signature` — observed metric values at failure (lets future runs detect recurrence).
- `closed_branches` — pattern strings the Executor must avoid; matched literally.
