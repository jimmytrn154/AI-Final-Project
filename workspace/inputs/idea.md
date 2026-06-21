# idea.md

# Learning When to Stop: Budget-Aware Early Exiting for Energy-Efficient CNN Inference

## Problem Statement

Modern convolutional neural networks typically perform the same full forward pass for every image, even though many inputs are easy enough to classify using earlier features. This wastes computation, increases latency, and contributes to unnecessary energy usage. This project studies whether an early-exit ResNet-18 can make input-dependent stopping decisions on CIFAR-10 to reduce inference cost while preserving most of the baseline accuracy.

## Core Hypothesis

A budget-aware early-exit CNN can reduce average FLOPs, latency, and estimated CO2 emissions compared with full ResNet-18 inference, while maintaining acceptable classification accuracy. The strongest policies are expected to balance confidence-based reliability with computation-aware stopping rather than always exiting as early as possible.

## 1. One-Sentence Summary

This project studies whether a ResNet-18 image classifier can learn input-dependent stopping decisions on CIFAR-10 so that easy samples exit early, reducing FLOPs, latency, and estimated carbon emissions while preserving most of the accuracy of full inference.

---

## 2. Research Problem

Standard CNN inference usually applies the same full forward pass to every input, even though some images are easier to classify than others. This wastes computation on samples that may already be confidently classified at intermediate layers.

This project reframes CNN inference as a budget-aware decision problem:

> For each input image, should the model stop at the current exit or continue to deeper layers?

The goal is not to permanently prune the network. Instead, the goal is dynamic, input-dependent inference where each image can use a different amount of computation.

---

## 3. Motivation

Efficient inference is important for Green AI and practical deployment because reducing unnecessary computation can lower:

- average FLOPs per sample,
- inference latency,
- estimated energy use and carbon emissions,
- and computational cost during deployment.

The project is motivated by the idea that a model should spend more computation only when the input is difficult enough to justify it.

---

## 4. Base Algorithm

The base algorithm is a ResNet-18 classifier trained on CIFAR-10.

The baseline performs full inference for every input:

```text
Input image
→ ResNet stem
→ Layer 1
→ Layer 2
→ Layer 3
→ Layer 4
→ Final classifier
```

The proposed early-exit model modifies this architecture by adding intermediate classifiers after selected ResNet stages:

```text
Input image
→ ResNet stem
→ Layer 1 → Exit 1
→ Layer 2 → Exit 2
→ Layer 3 → Exit 3
→ Layer 4 → Final exit
```

Each exit produces logits, class probabilities, a predicted label, and confidence statistics.

---

## 5. Core Research Question

The main research question is:

> Can budget-aware early exiting reduce inference cost, latency, and estimated emissions while maintaining acceptable CIFAR-10 classification accuracy?

A secondary question is:

> Can a learned controller, especially a REINFORCE-trained controller, make better exit decisions than manually designed fixed or dynamic confidence thresholds?

---

## 6. Proposed Variations

The project compares the full ResNet-18 baseline against several early-exit policies.

### 6.1 Early-Exit ResNet-18

The network is augmented with intermediate classifiers. All exits are trained jointly using weighted cross-entropy:

```math
L = \alpha_1 L_1 + \alpha_2 L_2 + \alpha_3 L_3 + \alpha_4 L_4
```

The final exit receives the largest weight to preserve full-model accuracy while allowing earlier exits to learn useful predictions.

### 6.2 Fixed Confidence Thresholding

At each exit, the model stops if the maximum softmax confidence exceeds a fixed threshold:

```math
\text{exit at } i \text{ if } c_i \geq \tau
```

This provides a simple hand-designed baseline for adaptive inference.

### 6.3 Budget-Aware Dynamic Thresholding

Instead of using one constant threshold, the exit threshold changes based on the fraction of computation already used:

```math
r_i = \frac{\text{FLOPs used up to exit } i}{\text{FLOPs of full ResNet-18}}
```

Two types of dynamic rules are considered:

- accuracy-first dynamic thresholding,
- budget-first dynamic thresholding.

These policies test whether computation awareness improves the accuracy-efficiency tradeoff.

### 6.4 Supervised Learned Exit Controller

A small controller is trained to decide whether to exit or continue. Its state features include:

```math
s_i = [c_i, H_i, m_i, i, r_i]
```

where:

- `c_i` is maximum softmax confidence,
- `H_i` is predictive entropy,
- `m_i` is top-1/top-2 probability margin,
- `i` is the current exit index,
- `r_i` is normalized FLOPs used so far.

The supervised controller is trained using labels such as earliest correct exit or reward-based target exit.

### 6.5 REINFORCE Exit Controller

The REINFORCE controller treats early exiting as a sequential decision-making problem. Each image is a short episode. At each non-final exit, the controller chooses:

```text
continue
exit
```

The reward balances correctness and computation:

```math
R = \mathbb{1}[\hat{y}=y] - \lambda r_i
```

where `r_i` is the normalized FLOPs ratio at the selected exit. This directly optimizes the accuracy-compute tradeoff.

---

## 7. Experimental Setting

### Dataset

CIFAR-10

### Task

10-class image classification.

### Framework

PyTorch.

### Efficiency Measurement

The project evaluates:

- test accuracy,
- average FLOPs per sample,
- FLOPs reduction relative to full ResNet-18,
- latency per sample,
- estimated CO2 emissions,
- average exit depth,
- exit distribution,
- and reward for controller-based policies.

---

## 8. Main Contributions

This project makes four main contributions:

1. **Early-exit ResNet-18 implementation for CIFAR-10**  
   The project modifies ResNet-18 with multiple intermediate exits and trains the exits jointly.

2. **Comparison of manual and learned exit policies**  
   The project compares fixed thresholding, budget-aware dynamic thresholding, supervised learned control, and REINFORCE-based control.

3. **Accuracy-efficiency analysis**  
   The project evaluates not only accuracy, but also FLOPs, latency, estimated emissions, exit distribution, and average exit depth.

4. **Green AI framing of adaptive inference**  
   The project frames early exiting as a compute-budget decision problem rather than static pruning.

---

## 9. Current Findings to Support in the Report

The current experiments show that adaptive inference can substantially reduce computation while preserving most of the baseline accuracy.

From the current progress tracker:

- Full ResNet-18 reaches 88.26% accuracy with 557.89M FLOPs/sample.
- The best fixed-threshold policy uses `tau = 0.95`, reaching 87.53% accuracy with 42.73% FLOPs reduction.
- The best dynamic-threshold policy reaches 87.41% accuracy with 35.48% FLOPs reduction.
- The supervised learned controller is very efficient but too aggressive, reaching 73.64% accuracy with 72.75% FLOPs reduction.
- The REINFORCE controller reaches 87.18% accuracy with 55.48% FLOPs reduction and average exit 2.07.

These findings should be verified against `experimental_log.md` and the actual CSV/JSON metrics before being stated in the final report.

---

## 10. Expected Paper Thesis

The expected thesis of the final report is:

> Early-exit inference provides a practical way to reduce CNN inference cost on CIFAR-10. Fixed thresholding gives the strongest observed accuracy-efficiency balance, while REINFORCE improves over the supervised learned controller by recovering most of the accuracy with stronger compute reduction. However, the learned policies require careful reward design and validation, and the current results should be interpreted within the limits of CIFAR-10 and the available experimental runs.

---

## 11. Important Limitations

The report should discuss the following limitations:

- The experiments are limited to CIFAR-10.
- No adaptive method currently surpasses the full ResNet-18 baseline accuracy.
- Some latency and emission numbers are baseline-scaled estimates rather than direct measurements.
- The supervised learned controller exits too aggressively.
- The REINFORCE controller improves over the supervised controller but does not dominate the best fixed-threshold policy in accuracy.
- Statistical confidence may be limited if multiple-seed results are unavailable.
- Results may not generalize directly to larger datasets or edge-device hardware.

---

## 12. Source-of-Truth Files

Use these files as the source of truth when writing the paper:

- `workspace/inputs/experimental_log.md`
- `outputs/metrics/comparison.csv`
- `outputs/metrics/baseline_metrics.json`
- `outputs/metrics/early_exit_metrics.json`
- `outputs/metrics/fixed_threshold_results.json`
- `outputs/metrics/dynamic_threshold_results.json`
- `outputs/metrics/controller_results.json`
- `outputs/metrics/reinforce_controller_results.json`
- `outputs/plots/accuracy_comparison.png`
- `outputs/plots/accuracy_vs_flops.png`
- `outputs/plots/accuracy_vs_latency.png`
- `outputs/plots/co2_reduction.png`
- `PROGRESS.md`
- `README.md`
- `REINFORCE_PHASE_PLAN.md`

Do not invent unsupported claims. If a number is missing from the metrics files, write `TODO: verify` instead of guessing.

---

## 13. Suggested Title Options

Primary title:

> Learning When to Stop: Budget-Aware Early Exiting for Energy-Efficient CNN Inference

Alternative titles:

1. Budget-Aware Early Exiting for Green CNN Inference
2. Learning Adaptive Stopping Policies for Efficient CIFAR-10 Classification
3. Accuracy-Efficiency Tradeoffs in Early-Exit ResNet Inference
4. Policy-Guided Early Exiting for Energy-Efficient Image Classification
