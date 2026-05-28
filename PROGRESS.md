# AI Final Project Progress Tracker

**Project:** Learning When to Stop: Budget-Aware Early Exiting for Energy-Efficient CNN Inference  
**Target Deadline:** July 12, 2026  
**Last Updated:** May 28, 2026

## Overall Status Summary
- Current stage: early-exit model trained; threshold experiments are next.
- Baseline and early-exit checkpoints/metrics are available.
- Result tables now contain baseline and jointly trained early-exit reference rows for future policy comparisons.

## Milestone Checklist
- [x] Environment and repo structure ready
- [x] CIFAR-10 loader works
- [x] Full ResNet-18 baseline trained
- [x] Baseline metrics recorded
- [x] Early-exit ResNet-18 implemented
- [x] Joint-loss training verified
- [x] Per-exit metrics recorded
- [ ] Fixed-threshold sweep completed
- [ ] Dynamic-threshold experiments completed
- [ ] Learned controller trained and evaluated
- [ ] Final plots and tables generated
- [ ] Report/package finalized
- [ ] Optional REINFORCE experiment completed

## Current Focus
- Run fixed-threshold sweep.
- Run dynamic-threshold experiments.
- Compare policy accuracy and FLOPs reduction against the baseline and trained early-exit model.

## Completed Tasks Log
- 2026-05-27: Reviewed `README.md` and aligned implementation order with the project phases.
- 2026-05-27: Created the project scaffold under `src/`, `scripts/`, `configs/`, `tests/`, `outputs/`, `checkpoints/`, and `data/`.
- 2026-05-27: Added baseline, early-exit, threshold, dynamic-threshold, controller, and plotting entrypoint scripts.
- 2026-05-27: Added this progress tracker as the working status document.
- 2026-05-28: Successfully trained the full ResNet-18 baseline for 20 epochs and saved `checkpoints/baseline_best.pt`.
- 2026-05-28: Recorded baseline metrics in `outputs/metrics/baseline_metrics.json` and `outputs/metrics/comparison.csv`.
- 2026-05-28: Trained early-exit ResNet-18 for 25 epochs and saved `checkpoints/early_exit_best.pt`.
- 2026-05-28: Recorded early-exit metrics in `outputs/metrics/early_exit_metrics.json` and `outputs/metrics/comparison.csv`.

## Pending Tasks
- Run `scripts/run_threshold_sweep.py`.
- Run `scripts/run_dynamic_threshold.py`.
- Run `scripts/train_controller.py`.
- Run `scripts/make_report_assets.py`.
- Update milestone checkboxes after each completed phase.

## Key Experiment Results
| Method | Accuracy | FLOPs/sample | FLOPs Reduction | Latency/sample | CO2 Emissions | Avg Exit | Notes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Full ResNet-18 | 88.26% | 557.89M | 0% | 2.18 ms | 9.23e-05 kg | Final | Baseline completed; best val accuracy 88.44% |
| Early-Exit ResNet-18 | 87.74% | 558.01M | 0% | 2.61 ms | 1.49e-04 kg | N/A | Joint-loss training completed; per-exit test accuracies 65.53%, 84.65%, 88.05%, 87.74%; best val final accuracy 88.20% |
| Fixed threshold | TBD | TBD | TBD | TBD | TBD | TBD | Awaiting sweep |
| Dynamic threshold | TBD | TBD | TBD | TBD | TBD | TBD | Awaiting sweep |
| Learned exit controller | TBD | TBD | TBD | TBD | TBD | TBD | Awaiting controller run |

## Blockers and Risks
- The final early-exit accuracy is slightly below the baseline test accuracy of 88.26%, while exit 3 reached 88.05%.
- Dynamic and controller latency/emissions are currently estimated or left pending until direct benchmarking is added.

## Decisions and Assumptions
- `PROGRESS.md` is the primary working tracker; `extraPlan.md` remains a strategy reference.
- The supervised learned controller is treated as core scope; REINFORCE remains optional.
- All comparisons should reference the full ResNet-18 baseline first.
- FLOPs ratios for exits are initialized from architecture-level estimates and can be refined with profiling later.

## Next 3 Actions
1. Run `scripts/run_threshold_sweep.py`.
2. Run `scripts/run_dynamic_threshold.py`.
3. Update tracker with threshold results and choose candidates for controller training.
