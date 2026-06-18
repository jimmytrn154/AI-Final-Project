# AI Final Project Progress Tracker

**Project:** Learning When to Stop: Budget-Aware Early Exiting for Energy-Efficient CNN Inference  
**Target Deadline:** July 12, 2026  
**Last Updated:** June 18, 2026

## Overall Status Summary
- Current stage: REINFORCE experiment completed; report packaging and final write-up are next.
- Baseline, early-exit, threshold, supervised controller, and REINFORCE controller metrics are now available for direct comparison.
- The best overall accuracy-efficiency tradeoff remains the fixed-threshold policy at `tau = 0.95`, while the REINFORCE controller provides a stronger compute reduction with only a small additional accuracy drop.

## Milestone Checklist
- [x] Environment and repo structure ready
- [x] CIFAR-10 loader works
- [x] Full ResNet-18 baseline trained
- [x] Baseline metrics recorded
- [x] Early-exit ResNet-18 implemented
- [x] Joint-loss training verified
- [x] Per-exit metrics recorded
- [x] Fixed-threshold sweep completed
- [x] Dynamic-threshold experiments completed
- [x] Learned controller trained and evaluated
- [x] Final plots and tables generated
- [ ] Report/package finalized
- [x] Optional REINFORCE experiment completed

## Current Focus
- Consolidate the best policy tradeoffs for the final report discussion.
- Finalize the report/package using the generated plots and completed metrics table.
- Write up the REINFORCE comparison against fixed-threshold, dynamic-threshold, and supervised controller baselines.

## Completed Tasks Log
- 2026-05-27: Reviewed `README.md` and aligned implementation order with the project phases.
- 2026-05-27: Created the project scaffold under `src/`, `scripts/`, `configs/`, `tests/`, `outputs/`, `checkpoints/`, and `data/`.
- 2026-05-27: Added baseline, early-exit, threshold, dynamic-threshold, controller, and plotting entrypoint scripts.
- 2026-05-27: Added this progress tracker as the working status document.
- 2026-05-28: Successfully trained the full ResNet-18 baseline for 20 epochs and saved `checkpoints/baseline_best.pt`.
- 2026-05-28: Recorded baseline metrics in `outputs/metrics/baseline_metrics.json` and `outputs/metrics/comparison.csv`.
- 2026-05-28: Trained early-exit ResNet-18 for 25 epochs and saved `checkpoints/early_exit_best.pt`.
- 2026-05-28: Recorded early-exit metrics in `outputs/metrics/early_exit_metrics.json` and `outputs/metrics/comparison.csv`.
- 2026-06-13: Completed the fixed-threshold sweep and saved `outputs/metrics/fixed_threshold_results.json`.
- 2026-06-13: Completed the dynamic-threshold sweep and saved `outputs/metrics/dynamic_threshold_results.json`.
- 2026-06-14: Trained the learned controller and saved `checkpoints/controller_best.pt`.
- 2026-06-14: Recorded controller metrics in `outputs/metrics/controller_results.json` and `outputs/metrics/comparison.csv`.
- 2026-06-14: Backfilled estimated controller FLOPs, latency, and CO2 metrics from the baseline reference.
- 2026-06-14: Generated report plots in `outputs/plots/` using `scripts/make_report_assets.py`.
- 2026-06-18: Completed the REINFORCE controller experiment and saved `outputs/metrics/reinforce_controller_results.json`.
- 2026-06-18: Recorded REINFORCE controller metrics in `outputs/metrics/comparison.csv`.

## Pending Tasks
- Update milestone checkboxes after each completed phase.

## Key Experiment Results
| Method | Accuracy | FLOPs/sample | FLOPs Reduction | Latency/sample | CO2 Emissions | Avg Exit | Notes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Full ResNet-18 | 88.26% | 557.89M | 0% | 2.18 ms | 9.23e-05 kg | Final | Baseline completed; best val accuracy 88.44% |
| Early-Exit ResNet-18 | 87.74% | 558.01M | 0% | 2.61 ms | 1.49e-04 kg | N/A | Joint-loss training completed; per-exit test accuracies 65.53%, 84.65%, 88.05%, 87.74%; best val final accuracy 88.20% |
| Fixed threshold | 87.53% | 319.51M | 42.73% | 1.25 ms | 5.29e-05 kg | 2.51 | Best fixed result at `tau = 0.95`; exit distribution 10.94%, 50.99%, 14.44%, 23.63% |
| Dynamic threshold | 87.41% | 359.92M | 35.48% | 1.41 ms | 5.96e-05 kg | 2.72 | Best dynamic result from `accuracy_first` with `alpha = 0.30` |
| Learned exit controller | 73.64% | 152.01M | 72.75% | 0.59 ms | 2.52e-05 kg | 1.24 | `earliest_correct` controller exits at Exit 1 for 80.39% of samples; efficiency metrics are baseline-scaled estimates |
| REINFORCE controller | 87.18% | 248.37M | 55.48% | 0.97 ms | 4.11e-05 kg | 2.07 | `lambda_cost = 0.20`; exit distribution 11.71%, 70.38%, 17.27%, 0.64%; average reward 0.783 |

## Blockers and Risks
- None of the current threshold policies surpass the baseline accuracy of 88.26%; the best fixed-threshold policy reaches 87.53%.
- The dynamic-threshold grid underperforms the best fixed-threshold result on both accuracy and estimated FLOPs reduction.
- The learned controller is too aggressive in its current form, exiting at Exit 1 for 80.39% of samples and dropping accuracy to 73.64%.
- The REINFORCE controller improves substantially over the supervised controller, but still does not beat the best fixed-threshold result on accuracy.
- Threshold and controller latency/emissions are baseline-scaled estimates rather than direct measurements.
- Show the tau ablation study, reward function for RL training, show accuracy for val/test across baselines, try cifar
- Compute stats value, average accuracy, variance of accuracy, remsemble/randomly sample the dataset, 

## Decisions and Assumptions
- `PROGRESS.md` is the primary working tracker; `extraPlan.md` remains a strategy reference.
- The supervised learned controller is treated as core scope; the optional REINFORCE phase has now been completed.
- All comparisons should reference the full ResNet-18 baseline first.
- FLOPs ratios for exits are initialized from architecture-level estimates and can be refined with profiling later.

## Next 3 Actions
1. Update the report discussion with the REINFORCE controller results and the current best fixed-threshold tradeoff.
2. Finalize the report/package using the completed metrics table and generated plots.
3. Decide whether any extra ablation or cleanup is worth doing before submission.
