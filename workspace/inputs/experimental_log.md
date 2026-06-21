# Experimental Log

## 1. Experimental Setup

This project evaluates budget-aware early exiting for CNN inference on CIFAR-10. The baseline is a full ResNet-18 classifier. The adaptive variants use an early-exit ResNet-18 with intermediate classifiers and different exit policies: fixed confidence thresholding, budget-aware dynamic thresholding, a supervised learned controller, and a REINFORCE-trained controller.

The main source-of-truth run for the final report is:

- `outputs/runs/2026-06-20-deeper-experiments/`

The main source-of-truth metrics are:

- `outputs/runs/2026-06-20-deeper-experiments/metrics/comparison.csv`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/ablation_comparison.csv`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/controller_lambda_sweep_results.json`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/threshold_ablation_results.json`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/dynamic_threshold_ablation_results.json`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/classification_metrics.json`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/bootstrap_statistics.json`

All numeric claims must come from these files or from other metric files explicitly listed in this log.

## 1. Project Objective

This project studies efficient image classification using adaptive inference on CIFAR-10.

The project compares a standard full-inference baseline with early-exit and controller-based inference policies. The goal is to study whether adaptive inference can reduce computational cost, latency, FLOPs, and estimated CO2 impact while preserving classification accuracy.

Main research question:

> Can adaptive exit policies improve the accuracy-efficiency tradeoff compared with full baseline inference?

Do not invent results. All numeric claims must come from the metric files listed in this log.

---

## 2. Raw Numeric Data

The final numeric values should be extracted by the agent from the project output files, not invented manually.

The agent must inspect the following directories:

- `outputs/metrics/`
- `outputs/plots/`
- `outputs/runs/2026-06-20-deeper-experiments/metrics/`
- `outputs/runs/2026-06-20-deeper-experiments/run_manifest.json`

Priority order for numeric evidence:

1. `outputs/runs/2026-06-20-deeper-experiments/metrics/`
2. `outputs/runs/2026-06-20-deeper-experiments/run_manifest.json`
3. `outputs/metrics/`
4. `PROGRESS.md`

If the same metric appears in multiple places and values conflict, prefer the highest-priority source above.

The agent should extract and report:

- accuracy
- FLOPs/sample
- FLOPs reduction
- latency/sample
- latency reduction, if available
- CO2 or energy metric
- CO2 reduction, if available
- average exit
- exit distribution
- reward, if applicable
- threshold or lambda setting, if applicable

The agent should build the final main result table from the most complete and reliable metric files.

Required output from the agent:

1. A filled main comparison table.
2. A short note explaining which metric files were used.
3. A note about any missing, conflicting, or estimated values.
4. A clear distinction between directly measured values and estimated/scaled values.

Do not invent missing values. If a value is unavailable, write `TODO: verify` or `N/A`.
### Dataset

CIFAR-10

### Task

10-class image classification.

### Dataset location

* `data/cifar-10-batches-py/`
* `data/cifar-10-python.tar.gz`

### Why this dataset is used

CIFAR-10 is used because it is a standard image classification benchmark that is small enough for repeated experimentation while still allowing comparison between baseline and adaptive inference methods.

---

## 3. Source-of-Truth Run

The main source-of-truth run for the final report is:

* `outputs/runs/2026-06-20-deeper-experiments/`

This run should be prioritized because it contains deeper experiments, ablations, controller lambda sweeps, and threshold ablation results.

Important files:

* `outputs/runs/2026-06-20-deeper-experiments/run_manifest.json`
* `outputs/runs/2026-06-20-deeper-experiments/metrics/comparison.csv`
* `outputs/runs/2026-06-20-deeper-experiments/metrics/ablation_comparison.csv`
* `outputs/runs/2026-06-20-deeper-experiments/metrics/baseline_metrics.json`
* `outputs/runs/2026-06-20-deeper-experiments/metrics/early_exit_metrics.json`
* `outputs/runs/2026-06-20-deeper-experiments/metrics/controller_lambda_sweep_results.json`
* `outputs/runs/2026-06-20-deeper-experiments/metrics/dynamic_threshold_ablation_results.json`
* `outputs/runs/2026-06-20-deeper-experiments/metrics/threshold_ablation_results.json`
* `outputs/runs/2026-06-20-deeper-experiments/metrics/classification_metrics.json`
* `outputs/runs/2026-06-20-deeper-experiments/metrics/bootstrap_statistics.json`

The earlier run may be used only as supporting evidence:

* `outputs/runs/2026-06-19-direct-latency/`

Do not let the earlier run override the source-of-truth run unless explicitly stated.

---

## 4. Methods Compared

| Method               | Description                                      | Config                              | Main evidence                                    |
| -------------------- | ------------------------------------------------ | ----------------------------------- | ------------------------------------------------ |
| Baseline             | Full-inference classifier without adaptive exits | `configs/baseline.json`             | `baseline_metrics.json`, `comparison.csv`        |
| Early Exit           | Classifier with intermediate exits               | `configs/early_exit.json`           | `early_exit_metrics.json`, `comparison.csv`      |
| Fixed Threshold      | Exit policy using a fixed confidence threshold   | `configs/fixed_threshold.json`      | `threshold_ablation_results.json`                |
| Dynamic Threshold    | Exit policy using adaptive thresholding          | `configs/dynamic_threshold.json`    | `dynamic_threshold_ablation_results.json`        |
| Learned Controller   | Learned policy for exit/continue decisions       | `configs/controller.json`           | `controller_lambda_sweep_results.json`           |
| REINFORCE Controller | Policy-gradient controller for adaptive exiting  | `configs/reinforce_controller.json` | `reinforce_controller_results.json` if available |

---

## 5. Metric File Registry

### Main comparison

Use this file for the main results table:

* `outputs/runs/2026-06-20-deeper-experiments/metrics/comparison.csv`

Expected columns to inspect:

* method name
* accuracy
* latency
* FLOPs
* CO2 or energy metric
* exit rate or average exit depth, if available

If column names differ, preserve the original column names and document them here.

### Ablation comparison

Use this file for the ablation table:

* `outputs/runs/2026-06-20-deeper-experiments/metrics/ablation_comparison.csv`

### Threshold ablation

Use this file for threshold sensitivity:

* `outputs/runs/2026-06-20-deeper-experiments/metrics/threshold_ablation_results.json`

### Dynamic threshold ablation

Use this file for dynamic threshold analysis:

* `outputs/runs/2026-06-20-deeper-experiments/metrics/dynamic_threshold_ablation_results.json`

### Controller lambda sweep

Use this file for reward/lambda sensitivity:

* `outputs/runs/2026-06-20-deeper-experiments/metrics/controller_lambda_sweep_results.json`

---

## 6. Existing Figure Registry

Existing plots are available in:

* `outputs/plots/accuracy_comparison.png`
* `outputs/plots/accuracy_vs_flops.png`
* `outputs/plots/accuracy_vs_latency.png`
* `outputs/plots/co2_reduction.png`

These should be copied or refined into:

* `workspace/inputs/figures/accuracy_comparison.png`
* `workspace/inputs/figures/accuracy_vs_flops.png`
* `workspace/inputs/figures/accuracy_vs_latency.png`
* `workspace/inputs/figures/co2_reduction.png`

The plotting agent should audit these figures before regenerating them.

Figure audit criteria:

* readable text
* clear axis labels
* correct units
* no misleading scale
* consistent method names
* values match the source metric files
* suitable size for LaTeX report
* caption can be supported by experimental evidence

---

## 7. Main Results Table

The final paper should include a main comparison table.

Build this table from:

* `outputs/runs/2026-06-20-deeper-experiments/metrics/comparison.csv`

Template:

| Method               |                Accuracy | Latency | FLOPs | CO2 / Energy Metric | Notes                       |
| -------------------- | ----------------------: | ------: | ----: | ------------------: | --------------------------- |
| Baseline             | TODO: fill from metrics |    TODO |  TODO |                TODO | Full inference              |
| Early Exit           | TODO: fill from metrics |    TODO |  TODO |                TODO | Intermediate exits          |
| Fixed Threshold      | TODO: fill from metrics |    TODO |  TODO |                TODO | Static confidence threshold |
| Dynamic Threshold    | TODO: fill from metrics |    TODO |  TODO |                TODO | Adaptive threshold          |
| Learned Controller   | TODO: fill from metrics |    TODO |  TODO |                TODO | Learned exit policy         |
| REINFORCE Controller | TODO: fill from metrics |    TODO |  TODO |                TODO | Policy-gradient exit policy |

Do not fill TODO values manually unless they are directly read from the metrics files.

---

## 8. Ablation Studies

### 8.1 Threshold Ablation

Evidence:

* `outputs/runs/2026-06-20-deeper-experiments/metrics/threshold_ablation_results.json`

Purpose:

Study how different confidence thresholds affect accuracy, exit behavior, latency, FLOPs, and efficiency.

Allowed claim style:

> Lower thresholds generally encourage earlier exits, while higher thresholds require more confidence before exiting.

Only include exact quantitative claims if supported by the JSON file.

### 8.2 Dynamic Threshold Ablation

Evidence:

* `outputs/runs/2026-06-20-deeper-experiments/metrics/dynamic_threshold_ablation_results.json`

Purpose:

Study whether dynamic thresholding improves the accuracy-efficiency tradeoff compared with fixed thresholding.

### 8.3 Controller Lambda Sweep

Evidence:

* `outputs/runs/2026-06-20-deeper-experiments/metrics/controller_lambda_sweep_results.json`

Purpose:

Study how reward tradeoff parameter lambda changes the behavior of the learned controller.

Allowed claim style:

> The best lambda should be selected from the sweep results, not assumed before reading the file.

---

## 9. Statistical Analysis Status

The assignment recommends statistical analysis using multiple runs so that mean and variance can be studied.

Current status:

* Multiple full training runs: TODO: verify
* Multiple seeds: TODO: verify
* Mean and variance reported: TODO: verify
* Statistical robustness script: `scripts/analyze_statistical_robustness.py`

If multiple runs or multiple seeds are unavailable, the report must explicitly justify this limitation.

Allowed wording if only one full run is available:

> Because of compute and time constraints, the final evaluation reports one main run and supports conclusions with ablation sweeps. The results should therefore be interpreted as an empirical case study rather than a statistically conclusive benchmark.

Do not claim statistical significance unless the robustness analysis supports it.

---

## 10. Figure Plan

The paper should include the following figures.

| Figure                  | Source                                                      | Action              | Purpose                              |
| ----------------------- | ----------------------------------------------------------- | ------------------- | ------------------------------------ |
| Accuracy comparison     | `outputs/plots/accuracy_comparison.png` or `comparison.csv` | audit/refine        | Compare accuracy across methods      |
| Accuracy vs FLOPs       | `outputs/plots/accuracy_vs_flops.png` or `comparison.csv`   | audit/refine        | Show accuracy-efficiency tradeoff    |
| Accuracy vs latency     | `outputs/plots/accuracy_vs_latency.png` or `comparison.csv` | audit/refine        | Show accuracy-latency tradeoff       |
| CO2 reduction           | `outputs/plots/co2_reduction.png` or `comparison.csv`       | audit/refine        | Show environmental/energy motivation |
| Threshold ablation      | `threshold_ablation_results.json`                           | generate if missing | Show threshold sensitivity           |
| Controller lambda sweep | `controller_lambda_sweep_results.json`                      | generate if missing | Show reward/lambda sensitivity       |

Figure generation rule:

Existing figures should be reused if they are accurate and publication-ready. Regenerate only if the existing figure is unreadable, inconsistent with the final metrics, missing labels/units, visually unsuitable for LaTeX, or not supported by the source-of-truth run.

---

## 11. Allowed Claims

The final report may claim the following only if supported by metrics:

1. Adaptive inference reduces FLOPs compared with full baseline inference.
2. Adaptive inference reduces latency compared with full baseline inference.
3. Adaptive inference reduces estimated CO2 or energy cost.
4. Early exits create an accuracy-efficiency tradeoff.
5. Threshold choice affects the tradeoff between accuracy and efficiency.
6. Learned controllers can optimize exit decisions using reward tradeoffs.
7. REINFORCE-based policies are promising but require careful tuning.

---

## 12. Disallowed Claims Unless Verified

Do not claim:

* A method is best unless the comparison table shows it.
* A method is statistically significant unless statistical analysis supports it.
* CO2 reduction is directly measured unless the metric was directly measured.
* REINFORCE is superior unless the results show that.
* The method generalizes to larger datasets unless tested.
* The method works on edge devices unless evaluated on edge-device hardware.

Use `TODO: verify` for unsupported claims.

---

## 13. Reproducibility Commands

Baseline:

```bash
bash run_baseline.sh
```

Early exit:

```bash
bash run_early_exit.sh
```

Fixed threshold:

```bash
bash run_fix_threshold.sh
```

Dynamic threshold:

```bash
bash run_dynamic_threshold.sh
```

Controller training:

```bash
bash run_train_controller.sh
```

Controller sweep:

```bash
bash run_controller_sweep.sh
```

REINFORCE training:

```bash
bash run_train_reinforce.sh
```

REINFORCE sweep:

```bash
bash run_reinforce_sweep.sh
```

Tests:

```bash
pytest
```

---

## 14. Limitations to Discuss

Possible limitations:

* Evaluation is limited to CIFAR-10.
* Results may not generalize to larger datasets.
* Latency may depend on hardware and measurement environment.
* CO2 estimates are approximate unless directly measured.
* Some methods may require more tuning.
* If only one run is available, statistical confidence is limited.
* Adaptive inference can introduce policy complexity compared with the baseline.
