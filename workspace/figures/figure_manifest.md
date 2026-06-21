# Figure Manifest

All figures below were audited from `workspace/inputs/figures/` and reused in `workspace/figures/` without regeneration. The deeper-experiment figures align with the source-of-truth run better than the older legacy plots in `outputs/plots/`, so the manuscript uses these newer files as the preferred visual evidence.

| File | Status | Used in paper | Evidence source | Audit note |
|---|---|---|---|---|
| `accuracy_flops_frontier.png` | Reused as-is | Yes | `ablation_comparison.csv` | Readable axes, correct method families, clear frontier view. |
| `accuracy_vs_threshold.png` | Reused as-is | Yes | `threshold_ablation_results.json` | Matches the monotonic threshold sweep and is publication-ready. |
| `controller_lambda_tradeoff.png` | Reused as-is | Yes | `controller_lambda_sweep_results.json` | Labels and trend are readable; suitable for the controller sweep discussion. |
| `reinforce_lambda_tradeoff.png` | Reused as-is | Yes | `reinforce_lambda_sweep_results.json` | Clear summary of the REINFORCE lambda tradeoff. |
| `exit_distribution_comparison.png` | Regenerated from metrics | Yes | `per_exit_analysis.json` | Re-rendered with shorter method labels and a cleaner stacked-bar layout for the final Results section. |
| `bootstrap_confidence_intervals.png` | Regenerated from metrics | Yes | `bootstrap_statistics.json` | Re-rendered with shorter labels and publication-style error bars so the uncertainty comparison is legible in the paper. |
| `confusion_matrix_best_methods.png` | Regenerated from metrics | Yes | `classification_metrics.json` | Re-rendered with concise panel titles while preserving the source confusion matrices. |

## Conflicts and Notes

- `experimental_log.md` still mentions legacy figure names such as `accuracy_vs_latency.png` and `co2_reduction.png`. Those files are not present in `workspace/inputs/figures/`, while the deeper-experiment figures are already available and consistent with the current source-of-truth run.
- The three additional-analysis figures were regenerated because their earlier versions used long raw method names that were too cramped for the final two-column report.
- Regeneration script: `workspace/figures/regenerate_additional_analysis.py`.
