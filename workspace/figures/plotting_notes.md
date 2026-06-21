# Plotting Notes

## Audit Outcome

- Existing figures in `workspace/inputs/figures/` were audited before any regeneration attempt.
- The audited figures were already based on the deeper experiment outputs and are consistent with the source-of-truth run `outputs/runs/2026-06-20-deeper-experiments/`.
- Three figures were regenerated in this pass to improve readability in the final two-column report:
  - `exit_distribution_comparison.png`
  - `bootstrap_confidence_intervals.png`
  - `confusion_matrix_best_methods.png`
- Regeneration script: `workspace/figures/regenerate_additional_analysis.py`

## Evidence Mapping

- `accuracy_flops_frontier.png` maps to `ablation_comparison.csv`.
- `accuracy_vs_threshold.png` maps to `threshold_ablation_results.json`.
- `controller_lambda_tradeoff.png` maps to `controller_lambda_sweep_results.json`.
- `reinforce_lambda_tradeoff.png` maps to `reinforce_lambda_sweep_results.json`.
- `bootstrap_confidence_intervals.png` maps to `bootstrap_statistics.json`.
- `exit_distribution_comparison.png` maps to `per_exit_analysis.json`.
- `confusion_matrix_best_methods.png` maps to `classification_metrics.json`.

## Source Conflicts Resolved

1. The newer deeper-experiment folder is the numeric source of truth, so legacy plots under `outputs/plots/` were not used for final numeric interpretation.
2. The run-specific `comparison.csv` is incomplete for policy rows. The manuscript therefore uses:
   - baseline and early-exit rows from `outputs/runs/2026-06-20-deeper-experiments/metrics/comparison.csv`
   - representative policy rows from the `best_result` entries in the family-specific JSON files
3. Policy `co2_kg` values are kept as FLOPs-scaled estimates. They should not be captioned or discussed as direct emissions measurements.
4. Policy latency values in the deeper run are direct staged-inference measurements, while the baseline latency comes from the reference comparison file. The paper notes this so the reader does not over-interpret absolute latency gaps.

## Main Draft Figure Selection

The current paper draft includes:

- `accuracy_flops_frontier.png`
- `accuracy_vs_threshold.png`
- `controller_lambda_tradeoff.png`
- `reinforce_lambda_tradeoff.png`
- `exit_distribution_comparison.png`
- `bootstrap_confidence_intervals.png`
- `confusion_matrix_best_methods.png`

## Added Tables

The strengthened Results section also adds two compact evidence-grounded tables derived from the deeper metrics:

- `tab:exit-behavior` from `per_exit_analysis.json`
- `tab:bootstrap-summary` from `bootstrap_statistics.json`
- `tab:classification-quality` from `classification_metrics.json`

Macro-F1, weighted-F1, macro precision, and macro recall were surfaced in a table rather than a new figure. The deltas between the strongest methods are small, and a compact table communicates the exact values more clearly than another bar chart.
