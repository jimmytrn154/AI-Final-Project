# AI Final Project Progress Tracker

**Project:** Learning When to Stop: Budget-Aware Early Exiting for Energy-Efficient CNN Inference  
**Target Deadline:** July 12, 2026  
**Last Updated:** June 13, 2026

## Overall Status Summary
- Current stage: fixed and dynamic threshold experiments completed; learned controller training is next.
- Baseline, early-exit, and threshold experiment metrics are now available for direct comparison.
- The current best policy is the fixed-threshold sweep at `tau = 0.95`, which preserves 87.53% accuracy while reducing estimated FLOPs by 42.73%.

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
- [ ] Learned controller trained and evaluated
- [ ] Final plots and tables generated
- [ ] Report/package finalized
- [ ] Optional REINFORCE experiment completed

## Current Focus
- Train the learned exit controller using the saved early-exit checkpoint.
- Generate final plots and tables from baseline, early-exit, fixed-threshold, and dynamic-threshold results.
- Consolidate the best policy tradeoffs for the final report discussion.

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

## Pending Tasks
- Run `scripts/train_controller.py`.
- Run `scripts/make_report_assets.py`.
- Update milestone checkboxes after each completed phase.

## Key Experiment Results
| Method | Accuracy | FLOPs/sample | FLOPs Reduction | Latency/sample | CO2 Emissions | Avg Exit | Notes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Full ResNet-18 | 88.26% | 557.89M | 0% | 2.18 ms | 9.23e-05 kg | Final | Baseline completed; best val accuracy 88.44% |
| Early-Exit ResNet-18 | 87.74% | 558.01M | 0% | 2.61 ms | 1.49e-04 kg | N/A | Joint-loss training completed; per-exit test accuracies 65.53%, 84.65%, 88.05%, 87.74%; best val final accuracy 88.20% |
| Fixed threshold | 87.53% | 319.51M | 42.73% | 1.25 ms | 5.29e-05 kg | 2.51 | Best fixed result at `tau = 0.95`; exit distribution 10.94%, 50.99%, 14.44%, 23.63% |
| Dynamic threshold | 87.41% | 359.92M | 35.48% | 1.41 ms | 5.96e-05 kg | 2.72 | Best dynamic result from `accuracy_first` with `alpha = 0.30` |
| Learned exit controller | TBD | TBD | TBD | TBD | TBD | TBD | Awaiting controller run |

## Blockers and Risks
- None of the current threshold policies surpass the baseline accuracy of 88.26%; the best fixed-threshold policy reaches 87.53%.
- The dynamic-threshold grid underperforms the best fixed-threshold result on both accuracy and estimated FLOPs reduction.
- Dynamic and controller latency/emissions are currently estimated or left pending until direct benchmarking is added.

## Decisions and Assumptions
- `PROGRESS.md` is the primary working tracker; `extraPlan.md` remains a strategy reference.
- The supervised learned controller is treated as core scope; REINFORCE remains optional.
- All comparisons should reference the full ResNet-18 baseline first.
- FLOPs ratios for exits are initialized from architecture-level estimates and can be refined with profiling later.

## Next 3 Actions
1. Run `scripts/train_controller.py`.
2. Run `scripts/make_report_assets.py`.
3. Update the report discussion and tracker with controller results, then decide whether the optional REINFORCE experiment is still worth pursuing.
