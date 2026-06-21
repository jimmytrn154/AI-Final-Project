# Research Brief

## Project Frame

- Title: Learning When to Stop: Budget-Aware Early Exiting for Energy-Efficient CNN Inference
- Dataset: CIFAR-10
- Backbone: ResNet-18
- Main question: can adaptive exit policies improve the accuracy-efficiency tradeoff over full inference?

## Source-of-Truth Evidence

Priority order taken from `workspace/inputs/experimental_log.md`:

1. `outputs/runs/2026-06-20-deeper-experiments/metrics/`
2. `outputs/runs/2026-06-20-deeper-experiments/run_manifest.json`
3. `outputs/metrics/`
4. `PROGRESS.md`

Primary files used for the manuscript:

- `outputs/runs/2026-06-20-deeper-experiments/metrics/comparison.csv`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/ablation_comparison.csv`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/threshold_ablation_results.json`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/dynamic_threshold_ablation_results.json`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/controller_lambda_sweep_results.json`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/reinforce_lambda_sweep_results.json`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/classification_metrics.json`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/bootstrap_statistics.json`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/per_exit_analysis.json`

## Main Quantitative Story

- Full ResNet-18 baseline: 88.26% accuracy, 557.89M FLOPs/sample, 2.182 ms latency.
- Early-exit backbone without a policy: 87.74% final-exit accuracy, 558.01M FLOPs/sample, 2.606 ms latency.
- Best fixed-threshold accuracy point: `tau = 0.99`, 87.73% accuracy, 32.83% FLOPs reduction, average exit 2.88.
- Best dynamic-threshold accuracy point: `accuracy_first`, base threshold 0.90, `alpha = 0.50`, 87.74% accuracy and 0% FLOPs reduction because all samples use the final exit.
- Best supervised `best_reward` controller: `lambda = 0.30`, 72.52% accuracy, 74.33% FLOPs reduction, average exit 1.18.
- Best REINFORCE run by accuracy: `lambda_cost = 0.05`, 87.69% accuracy, 49.54% FLOPs reduction, average exit 2.32.

## Evidence Caveats To Preserve

- The run-specific `comparison.csv` contains only the baseline and early-exit rows.
- Representative rows for fixed threshold, dynamic threshold, supervised controller, and REINFORCE therefore come from the `best_result` entries in their family-specific JSON files.
- Baseline and early-exit `co2_kg` values come from CodeCarbon-wrapped training runs.
- Policy `co2_kg` values are FLOPs-scaled estimates and should not be described as directly measured emissions.
- Policy latency values in the deeper experiment run are direct staged-inference measurements, while the baseline latency is inherited from the reference comparison file. The paper should treat absolute latency gaps cautiously and describe the benchmarking paths honestly.

## Figure Plan

- Reuse `accuracy_flops_frontier.png` as the main tradeoff figure.
- Reuse `accuracy_vs_threshold.png` for fixed-threshold sensitivity.
- Reuse `controller_lambda_tradeoff.png` for the supervised `best_reward` sweep.
- Reuse `reinforce_lambda_tradeoff.png` for the REINFORCE sweep.
- Keep `bootstrap_confidence_intervals.png`, `exit_distribution_comparison.png`, and `confusion_matrix_best_methods.png` as audited supporting figures.
