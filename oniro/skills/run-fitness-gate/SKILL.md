---
name: run-fitness-gate
description: Execute the empirical Gödel-relaxation gate. Returns ACCEPT / REJECT / UNDECIDED with structured notes.
trigger: orchestrator-after-mutation
allowed_tools: [bash, python]
output: json {verdict, splits_improved, mean_delta, sigma_noise, threshold, notes}
---

# run-fitness-gate

You orchestrate the empirical Gödel relaxation. The gate logic is in
`oniro/orchestrator/godel_gate.py`; this skill drives the I/O around it.

## Procedure

1. Load `theta_0` (parent checkpoint) and `theta_1` (candidate after mutation).
2. For each of the 10 OOD splits in `oniro/eval/ood_buffer/`:
   - Run a fixed-seed forward pass with `theta_0` and record predictive loss.
   - Run the same forward pass with `theta_1` and record predictive loss.
3. Call `GodelGate.evaluate(baseline_losses, candidate_losses)`.
4. Persist the `GateDecision` to `oniro/wiki/variants/<variant_id>.md` under the
   `gate:` frontmatter field.
5. If `verdict == "UNDECIDED"`, schedule two additional seeds and re-call.

## Rules

- **Never** evaluate the candidate on the same data it was just trained on.
- **Never** lower the threshold to pass a borderline case; that defeats the gate.
- **Always** refresh `sigma_noise` weekly via a no-op control run (same data,
  different seed, no mutation).

## Output

Direct JSON copy of `GateDecision.__dict__`:

```json
{
  "verdict": "ACCEPT",
  "splits_improved": 9,
  "mean_delta": 0.0182,
  "sigma_noise": 0.0091,
  "threshold": 0.00455,
  "notes": "9/10 improved, mean_delta=0.0182 > 0.00455"
}
```
