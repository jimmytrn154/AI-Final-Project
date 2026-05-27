# AI Final Project Progress Tracker

**Project:** Learning When to Stop: Budget-Aware Early Exiting for Energy-Efficient CNN Inference  
**Target Deadline:** July 12, 2026  
**Last Updated:** May 27, 2026

## Overall Status Summary
- Current stage: project scaffold and experiment pipeline implemented.
- Training and experiment execution are not complete yet.
- Result tables are ready to receive outputs from baseline, early-exit, threshold, dynamic, and controller runs.

## Milestone Checklist
- [x] Environment and repo structure ready
- [ ] CIFAR-10 loader works
- [ ] Full ResNet-18 baseline trained
- [ ] Baseline metrics recorded
- [x] Early-exit ResNet-18 implemented
- [ ] Joint-loss training verified
- [ ] Per-exit metrics recorded
- [ ] Fixed-threshold sweep completed
- [ ] Dynamic-threshold experiments completed
- [ ] Learned controller trained and evaluated
- [ ] Final plots and tables generated
- [ ] Report/package finalized
- [ ] Optional REINFORCE experiment completed

## Current Focus
- Verify the local environment has the required Python packages.
- Run the baseline training pipeline and save `baseline_metrics.json`.
- Train the early-exit model so the policy experiments can begin.

## Completed Tasks Log
- 2026-05-27: Reviewed `README.md` and aligned implementation order with the project phases.
- 2026-05-27: Created the project scaffold under `src/`, `scripts/`, `configs/`, `tests/`, `outputs/`, `checkpoints/`, and `data/`.
- 2026-05-27: Added baseline, early-exit, threshold, dynamic-threshold, controller, and plotting entrypoint scripts.
- 2026-05-27: Added this progress tracker as the working status document.

## Pending Tasks
- Install or confirm dependencies from `requirements.txt`.
- Run `scripts/train_baseline.py`.
- Run `scripts/train_early_exit.py`.
- Run `scripts/run_threshold_sweep.py`.
- Run `scripts/run_dynamic_threshold.py`.
- Run `scripts/train_controller.py`.
- Run `scripts/make_report_assets.py`.
- Update milestone checkboxes after each completed phase.

## Key Experiment Results
| Method | Accuracy | FLOPs/sample | FLOPs Reduction | Latency/sample | CO2 Emissions | Avg Exit | Notes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Full ResNet-18 | TBD | TBD | 0% | TBD | TBD | Final | Awaiting baseline run |
| Early-Exit ResNet-18 | TBD | TBD | TBD | TBD | TBD | N/A | Awaiting training |
| Fixed threshold | TBD | TBD | TBD | TBD | TBD | TBD | Awaiting sweep |
| Dynamic threshold | TBD | TBD | TBD | TBD | TBD | TBD | Awaiting sweep |
| Learned exit controller | TBD | TBD | TBD | TBD | TBD | TBD | Awaiting controller run |

## Blockers and Risks
- `torch`, `torchvision`, `thop`, `codecarbon`, or `matplotlib` may not be installed yet.
- CIFAR-10 download requires network access when the dataset is not already cached.
- Dynamic and controller latency/emissions are currently estimated or left pending until direct benchmarking is added.

## Decisions and Assumptions
- `PROGRESS.md` is the primary working tracker; `extraPlan.md` remains a strategy reference.
- The supervised learned controller is treated as core scope; REINFORCE remains optional.
- All comparisons should reference the full ResNet-18 baseline first.
- FLOPs ratios for exits are initialized from architecture-level estimates and can be refined with profiling later.

## Next 3 Actions
1. Confirm dependencies and run the baseline training script.
2. Train the early-exit model and verify per-exit metrics.
3. Execute fixed and dynamic threshold experiments, then update the tracker with real numbers.
