![Quote](card1.jpg)

# Learning When to Stop: Budget-Aware Early Exiting for Energy-Efficient CNN Inference

**Student:** Tran Anh Chuong  
**University:** VinUniversity  
**Course:** COMP2050 - AI Programming Project  
**Target Deadline:** July 22, 2026

---

## 1. Project Summary

This project studies input-dependent early exiting for CIFAR-10 classification with a ResNet-18 backbone. Instead of forcing every image through the full network, the model can stop at intermediate exits when a policy decides the prediction is already reliable enough.

The implemented comparison now includes:

- a full ResNet-18 baseline
- a jointly trained early-exit ResNet-18
- fixed-threshold early-exit policies
- dynamic-threshold early-exit policies
- a supervised learned exit controller
- a REINFORCE-based exit controller

The main goal is to compare the tradeoff between:

- accuracy
- average FLOPs
- latency
- approximate emissions reporting

---

## 2. Current Status

The repository is no longer just a project plan. The following are already implemented and runnable:

- CIFAR-10 data pipeline
- full ResNet-18 training and evaluation
- early-exit ResNet-18 with 3 intermediate exits plus the final exit
- joint-loss training for the early-exit backbone
- fixed-threshold and dynamic-threshold sweeps
- supervised controller training
- REINFORCE controller training
- plot generation from comparison metrics
- versioned rerun outputs under `outputs/runs/<run_name>/`

Current working tracker:

- [PROGRESS.md](/home/24chuong.ta/chun/AI-Final-Project/PROGRESS.md)

REINFORCE-specific implementation notes:

- [REINFORCE_PHASE_PLAN.md](/home/24chuong.ta/chun/AI-Final-Project/REINFORCE_PHASE_PLAN.md)

---

## 3. Model and Methods

### Backbone

- Dataset: CIFAR-10
- Base model: ResNet-18
- Framework: PyTorch

### Early-exit architecture

The early-exit model adds auxiliary classifiers after:

- Layer 1
- Layer 2
- Layer 3
- Layer 4 / final classifier

This gives 4 total exits:

- `exit_1`
- `exit_2`
- `exit_3`
- `final_exit`

### Implemented decision strategies

1. `Fixed threshold`
   Exit when max softmax confidence exceeds a constant threshold.

2. `Dynamic threshold`
   Use either `accuracy_first` or `budget_first` threshold schedules based on cumulative compute.

3. `Supervised controller`
   Train a small MLP controller using state features:
   confidence, normalized entropy, top-1/top-2 margin, normalized exit depth, and FLOPs ratio.

4. `REINFORCE controller`
   Fine-tune the controller with policy gradient, using a reward that balances correctness and compute cost.

---

## 4. Metrics and Important Caveats

The repo currently reports:

- accuracy
- FLOPs per sample
- FLOPs reduction
- latency
- latency reduction
- CO2 / emissions
- average exit
- exit distribution

### What is measured directly

`Accuracy`
- Measured directly from evaluation on the test set.

`FLOPs`
- Baseline and early-exit backbone FLOPs are profiled directly.
- Policy-method FLOPs are derived from average exit usage and the model's stored exit FLOPs ratios.

`Latency`
- Baseline and early-exit backbone latency are measured directly.
- In the `outputs/runs/2026-06-19-direct-latency/` rerun, threshold, controller, and REINFORCE latency are also measured directly using true conditional execution.

### What is not measured directly

`CO2 / emissions`
- Direct CodeCarbon measurement is currently used only for the two training backbones:
  - full ResNet-18 training
  - early-exit ResNet-18 training
- Threshold, dynamic-threshold, supervised-controller, and REINFORCE-controller CO2 values are **not** direct CodeCarbon measurements.
- Those policy CO2 values are estimated by scaling the baseline reference with the method's average FLOPs ratio.

The current policy-side formula is:

```text
estimated_co2_kg = baseline_co2_kg * average_flops_ratio
```

### Important interpretation note

The current CO2 numbers are useful as rough efficiency proxies, but they are **not a clean apples-to-apples inference-emissions benchmark** across all methods.

Why:

- the baseline and early-exit `co2_kg` values come from CodeCarbon-wrapped training runs
- the policy methods use baseline-scaled estimates instead of direct emissions measurement

So the safest interpretation today is:

- use `accuracy`, `FLOPs`, and `latency` as the primary comparison metrics
- treat `co2_kg` for non-training policies as an approximate proxy only

If direct inference-emissions comparison is needed later, the policy evaluation scripts will need their own dedicated CodeCarbon benchmarking pass.

---

## 5. Current Results Snapshot

The latest direct-latency rerun is stored under:

- [outputs/runs/2026-06-19-direct-latency/](/home/24chuong.ta/chun/AI-Final-Project/outputs/runs/2026-06-19-direct-latency)

The current comparison table is:

| Method | Accuracy | FLOPs/sample | FLOPs Reduction | Latency/sample | Avg Exit | Notes |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Full ResNet-18 | 88.26% | 557.89M | 0.00% | 2.182 ms | Final | Baseline reference |
| Early-Exit ResNet-18 | 87.74% | 558.01M | 0.00% | 2.606 ms | N/A | Joint-loss backbone only |
| Fixed threshold (`tau = 0.95`) | 87.53% | 319.51M | 42.73% | 0.075 ms | 2.51 | Best fixed-threshold result |
| Dynamic threshold (`accuracy_first`, `alpha = 0.30`) | 87.41% | 359.92M | 35.48% | 0.078 ms | 2.72 | Best dynamic-threshold result |
| Supervised controller | 73.64% | 152.01M | 72.75% | 0.077 ms | 1.24 | Very aggressive early exiting |
| REINFORCE controller (`lambda_cost = 0.20`) | 86.91% | 240.84M | 56.83% | 0.085 ms | 2.00 | Stronger tradeoff than supervised controller |

Primary artifacts:

- [comparison.csv](/home/24chuong.ta/chun/AI-Final-Project/outputs/runs/2026-06-19-direct-latency/metrics/comparison.csv)
- [fixed_threshold_results.json](/home/24chuong.ta/chun/AI-Final-Project/outputs/runs/2026-06-19-direct-latency/metrics/fixed_threshold_results.json)
- [dynamic_threshold_results.json](/home/24chuong.ta/chun/AI-Final-Project/outputs/runs/2026-06-19-direct-latency/metrics/dynamic_threshold_results.json)

Note: the `co2_kg` column still exists in the metrics files, but for threshold/controller/REINFORCE methods it remains baseline-scaled rather than directly measured.

---

## 6. Repository Layout

```text
AI-Final-Project/
├── checkpoints/
├── configs/
├── data/
├── outputs/
│   ├── metrics/
│   ├── plots/
│   └── runs/
├── scripts/
├── src/ai_final_project/
├── tests/
├── PROGRESS.md
├── README.md
└── REINFORCE_PHASE_PLAN.md
```

Important subdirectories:

- `src/ai_final_project/`
  Core model, training, evaluation, profiling, controller, and results utilities.
- `scripts/`
  Runnable experiment entrypoints.
- `configs/`
  JSON configs for baseline, early-exit, threshold, controller, and REINFORCE runs.
- `outputs/metrics/`
  Legacy metrics snapshot.
- `outputs/runs/<run_name>/`
  Versioned rerun artifacts so new experiments do not overwrite previous results.

---

## 7. Main Scripts

Training and evaluation entrypoints:

- `scripts/train_baseline.py`
- `scripts/train_early_exit.py`
- `scripts/run_threshold_sweep.py`
- `scripts/run_dynamic_threshold.py`
- `scripts/train_controller.py`
- `scripts/train_reinforce_controller.py`
- `scripts/make_report_assets.py`

Deeper ablation and analysis entrypoints:

- `scripts/run_threshold_ablation.py`
- `scripts/run_controller_lambda_sweep.py`
- `scripts/run_reinforce_sweep.py`
- `scripts/analyze_statistical_robustness.py`
- `scripts/make_ablation_assets.py`

Shell wrappers already present in the repo:

- `run_baseline.sh`
- `run_early_exit.sh`
- `run_fix_threshold.sh`
- `run_dynamic_threshold.sh`
- `run_train_controller.sh`
- `run_train_reinforce.sh`

---

## 8. Recommended Run Order

For a fresh full pipeline:

1. Train the full baseline
2. Train the early-exit backbone
3. Run the fixed-threshold sweep
4. Run the dynamic-threshold sweep
5. Train the supervised controller
6. Train the REINFORCE controller
7. Generate plots

Example Python commands:

```bash
python scripts/train_baseline.py --config configs/baseline.json
python scripts/train_early_exit.py --config configs/early_exit.json
python scripts/run_threshold_sweep.py --config configs/fixed_threshold.json
python scripts/run_dynamic_threshold.py --config configs/dynamic_threshold.json
python scripts/train_controller.py --config configs/controller.json
python scripts/train_reinforce_controller.py --config configs/reinforce_controller.json
python scripts/make_report_assets.py --config configs/baseline.json
```

For a versioned rerun with isolated outputs:

```bash
python scripts/run_threshold_sweep.py --config configs/fixed_threshold.json --run-name 2026-06-19-direct-latency
python scripts/run_dynamic_threshold.py --config configs/dynamic_threshold.json --run-name 2026-06-19-direct-latency
python scripts/train_controller.py --config configs/controller.json --run-name 2026-06-19-direct-latency
python scripts/train_reinforce_controller.py --config configs/reinforce_controller.json --run-name 2026-06-19-direct-latency
python scripts/make_report_assets.py --config configs/baseline.json --run-name 2026-06-19-direct-latency
```

Important rerun note:

- When `--run-name` is used, controller checkpoints are written into that run's own `checkpoints/` folder.
- The REINFORCE run expects the supervised controller warm-start checkpoint inside the same run folder, so the supervised controller should be run before REINFORCE in that versioned run.

---

## 9. Outputs and Artifacts

### Legacy outputs

- `outputs/metrics/`
- `outputs/plots/`

These contain the original saved metrics and report figures.

### Versioned rerun outputs

Each rerun can now be stored under:

```text
outputs/runs/<run_name>/
  checkpoints/
  metrics/
  plots/
  run_manifest.json
```

Run bootstrapping behavior:

- a fresh run copies `baseline_metrics.json` and `early_exit_metrics.json` from the reference metrics directory
- a fresh run seeds `comparison.csv` with only:
  - `full_resnet18`
  - `early_exit_resnet18`
- threshold/controller/REINFORCE rows are then added only for that run

This makes it easy to compare old and new experiment passes without overwriting previous metrics.

---

## 10. Tests

The repo currently includes tests for:

- model output shapes and staged early-exit routing
- threshold policy logic
- REINFORCE helper functions
- versioned output path resolution and run bootstrapping

Run tests with:

```bash
python -m unittest discover -s tests
```

---

## 11. What To Use For Analysis

For the current repo state, the most reliable interpretation is:

- `Accuracy`: direct and trustworthy
- `FLOPs`: strong comparison metric
- `Latency`: direct for the latest rerun under `outputs/runs/2026-06-19-direct-latency/`
- `CO2`: direct only for the two backbone training runs; estimated for the policy methods

So if you are writing the final report now:

- lead with accuracy, FLOPs, and latency
- clearly label policy CO2 as estimated
- avoid claiming that threshold/controller/REINFORCE emissions were directly measured with CodeCarbon

---

## 12. Next Work

The remaining high-level work is:

- run the deeper CIFAR-10 ablation suite:
  - dense fixed/dynamic threshold sweeps
  - learned-controller `best_reward` lambda sweep
  - REINFORCE lambda sweep
  - bootstrap confidence intervals and richer classification metrics
- finalize the report and package
- write up the tradeoff discussion, especially:
  - fixed threshold vs dynamic threshold
  - supervised controller vs REINFORCE controller
  - direct latency gains vs still-estimated emissions
- optionally add a dedicated inference-time CodeCarbon benchmark in a future upgrade if direct emissions comparison is required
