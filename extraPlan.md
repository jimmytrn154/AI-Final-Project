![Quote](card1.jpg)

# Learning When to Stop: Budget-Aware Early Exiting for Energy-Efficient CNN Inference

**Student:** Tran Anh Chuong  
**University:** VinUniversity  
**Course:** COMP2050 - AI Programming Project  
**Target Deadline:** July 12, 2026  

---

## 1. Project Overview

Modern Convolutional Neural Networks (CNNs) usually perform the same full forward pass for every input, even though some images are much easier to classify than others. This wastes computation, increases inference latency, and contributes to unnecessary energy usage.

This project explores a **Green AI** approach using **budget-aware dynamic early exiting**. Instead of forcing every CIFAR-10 image to pass through the full ResNet-18 model, the network is modified with intermediate classifiers, called **early exits**, placed after selected ResNet stages. If an intermediate classifier is sufficiently reliable, the model stops early and returns the prediction. If not, the image continues to deeper layers.

The central idea is:

> The model should spend more computation only when the input is difficult enough to justify it.

This project is therefore not standard static pruning. It does not permanently remove weights or filters from the model. Instead, it performs **input-dependent inference**, where each image may use a different amount of computation.

---

## 2. Research Question

The main research question is:

> **Can a CNN learn when to stop computation in order to reduce FLOPs, latency, and carbon emissions while maintaining acceptable classification accuracy?**

To answer this, the project compares three main exit-decision strategies:

1. **Fixed confidence thresholding**  
   Exit when the current classifier's confidence is above a fixed threshold.

2. **Budget-aware dynamic thresholding**  
   Adjust the exit threshold based on how much computation has already been used.

3. **Learned exit controller**  
   Train a small controller to decide whether to exit or continue using confidence, entropy, margin, exit index, and FLOPs-used features.

An additional optional phase will explore **policy-gradient / REINFORCE training** for the learned controller after the main project pipeline is complete.

---

## 3. Base Algorithm and Environment

- **Backbone model:** ResNet-18
- **Task:** Image classification
- **Dataset:** CIFAR-10
- **Framework:** PyTorch
- **Efficiency tracking:** FLOPs, latency, and CodeCarbon emissions
- **Main comparison:** Full ResNet-18 vs early-exit ResNet-18 variants

---

## 4. Proposed Model: Early-Exit ResNet-18

The modified architecture adds auxiliary classifiers after intermediate ResNet stages:

```text
Input image
→ ResNet stem
→ Layer 1 → Exit 1
→ Layer 2 → Exit 2
→ Layer 3 → Exit 3
→ Layer 4 → Final exit
```

Each exit produces a class prediction and a confidence score. For exit `i`, the output probability is:

```math
p_i = softmax(z_i)
```

The confidence score is:

```math
c_i = \max_j p_i(j)
```

The goal is not to exit at the shallowest layer for every image. The goal is to find the **earliest reliable exit** for each input.

---

## 5. Training Objective

All exits are trained jointly using weighted cross-entropy loss:

```math
L = \alpha_1 L_1 + \alpha_2 L_2 + \alpha_3 L_3 + \alpha_4 L_4
```

A planned initial weighting scheme is:

```math
\alpha_1 = 0.3, \quad \alpha_2 = 0.5, \quad \alpha_3 = 0.7, \quad \alpha_4 = 1.0
```

The final classifier receives the largest weight to preserve the full model's accuracy while still allowing earlier exits to learn useful predictions.

---

## 6. Exit-Decision Strategies

### 6.1 Strategy 1: Fixed Confidence Thresholding

The simplest strategy exits when the maximum softmax confidence exceeds a fixed threshold:

```math
\text{exit at } i \text{ if } c_i \geq \tau
```

Planned threshold sweep:

```text
τ = 0.60, 0.70, 0.80, 0.90, 0.95
```

Expected behavior:

- Lower threshold → more early exits, lower FLOPs, possible accuracy drop.
- Higher threshold → fewer early exits, higher accuracy, less compute saving.

### 6.2 Strategy 2: Budget-Aware Dynamic Thresholding

Instead of using one fixed threshold, the threshold changes based on the computation already used.

Let:

```math
r_i = \frac{\text{FLOPs used up to exit } i}{\text{FLOPs of full ResNet-18}}
```

Two variants will be tested.

**Accuracy-first thresholding:**

```math
\tau_i = \tau_0 + \alpha r_i
```

This becomes stricter as more computation is used.

**Budget-first thresholding:**

```math
\tau_i = \tau_0 - \alpha r_i
```

This becomes more willing to exit as the model approaches the full computation budget. This variant is especially aligned with the Green AI goal.

### 6.3 Strategy 3: Learned Exit Controller

The learned controller treats early exiting as a small decision-making problem. At each exit, it decides:

```math
a_i \in \{\text{exit}, \text{continue}\}
```

The controller receives the feature vector:

```math
s_i = [c_i, H_i, m_i, i, r_i]
```

where:

- `c_i`: maximum softmax confidence
- `H_i`: entropy of the prediction
- `m_i`: margin between top-1 and top-2 probabilities
- `i`: current exit index
- `r_i`: fraction of FLOPs already used

Entropy is:

```math
H_i = -\sum_j p_i(j) \log p_i(j)
```

Margin is:

```math
m_i = p_i(\text{top-1}) - p_i(\text{top-2})
```

The reward idea is:

```math
R = \mathbb{1}[\hat{y}=y] - \lambda \cdot \frac{\text{FLOPs}_i}{\text{FLOPs}_{full}}
```

This rewards correct predictions while penalizing expensive computation. In practice, the first implementation may train the controller using the **earliest correct exit** or **best reward exit** as a supervised target, then compare it with the reward-based version if time allows.

---

## 7. Evaluation Plan

The project will compare the full ResNet-18 baseline against the three early-exit strategies.

### 7.1 Main Metrics

- **Test accuracy (%):** Measures classification performance.
- **Average FLOPs/sample:** Measures computational cost.
- **FLOPs reduction (%):** Measures computation saved compared with full ResNet-18.
- **Inference latency (ms):** Measures real runtime speedup.
- **Carbon emissions (gCO₂eq):** Estimated using the `codecarbon` Python library.
- **Exit distribution:** Shows the percentage of samples exiting at Exit 1, Exit 2, Exit 3, and the final exit.

FLOPs reduction is computed as:

```math
\text{FLOPs Reduction} = 1 - \frac{\text{Average FLOPs}_{method}}{\text{FLOPs}_{full}}
```

### 7.2 Planned Result Tables

**Baseline table:**

| Model | Accuracy | FLOPs/sample | Latency/sample | CO₂ Emissions |
| :--- | ---: | ---: | ---: | ---: |
| Full ResNet-18 | TBD | TBD | TBD | TBD |

**Final comparison table:**

| Method | Accuracy | FLOPs Reduction | Latency Reduction | CO₂ Reduction | Avg Exit |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Full ResNet-18 | TBD | 0% | 0% | 0% | Final |
| Fixed threshold | TBD | TBD | TBD | TBD | TBD |
| Dynamic threshold | TBD | TBD | TBD | TBD | TBD |
| Learned exit controller | TBD | TBD | TBD | TBD | TBD |
| REINFORCE exit controller | Optional | Optional | Optional | Optional | Optional |

### 7.3 Planned Visualizations

- Accuracy vs FLOPs reduction curve
- Accuracy vs latency reduction curve
- Exit distribution bar chart
- Carbon emissions comparison
- Accuracy-efficiency frontier across all strategies

---

## 8. Implementation Roadmap

### Phase 1: Baseline Setup

Train a standard ResNet-18 on CIFAR-10 and build the evaluation pipeline.

Deliverables:

- Working CIFAR-10 data loader
- Full ResNet-18 training script
- Baseline test accuracy
- Baseline FLOPs, latency, and CodeCarbon measurement

### Phase 2: Early-Exit Architecture

Modify ResNet-18 by adding auxiliary classifiers after intermediate ResNet stages.

Deliverables:

- Early-exit ResNet-18 model
- Joint weighted-loss training
- Per-exit accuracy report

### Phase 3: Fixed Threshold Experiments

Run threshold sweeps using maximum softmax confidence.

Deliverables:

- Accuracy vs FLOPs curve
- Exit distribution for each threshold
- Best fixed-threshold configuration

### Phase 4: Budget-Aware Dynamic Thresholding

Test dynamic threshold rules that adjust based on used FLOPs.

Deliverables:

- Accuracy-first dynamic threshold results
- Budget-first dynamic threshold results
- Comparison with fixed thresholding

### Phase 5: Learned Exit Controller

Train a small controller using exit features and supervised reward-based labels.

Deliverables:

- Controller dataset from exit features
- Learned controller training script
- Comparison against fixed and dynamic thresholding

### Phase 6: Reporting and Finalization

Prepare the final report, visualizations, code package, and code-origin statement.

Deliverables:

- Final result tables
- Final plots
- LaTeX report
- `code.zip`
- `statement.pdf`

---

## 9. Optional Extra Phase: Policy-Gradient / REINFORCE Training

This phase is an **extra research extension** and should only be attempted after the main project pipeline is complete:

1. Full ResNet-18 baseline is trained and evaluated.
2. Early-exit ResNet-18 is trained successfully.
3. Fixed confidence thresholding experiments are completed.
4. Budget-aware dynamic thresholding experiments are completed.
5. The learned exit controller works using supervised or reward-labeled training.

The purpose of this extra phase is to make the learned exit controller more explicitly reinforcement-learning based.

### 9.1 Motivation

The learned exit controller can be viewed as a policy that decides whether to stop inference or continue deeper into the network. In the main implementation, the controller may be trained using supervised labels such as the **earliest correct exit** or the **best reward exit**. In this optional phase, the controller will instead be trained with a policy-gradient method.

At each exit `i`, the controller observes the state:

```math
s_i = [c_i, H_i, m_i, i, r_i]
```

Then it samples an action from its policy:

```math
a_i \sim \pi_\theta(a_i \mid s_i)
```

where:

- `a_i = exit`: stop inference and return the current prediction
- `a_i = continue`: continue to the next ResNet stage

This makes each image a short decision-making episode.

### 9.2 Reward Function

When the controller eventually exits, it receives the accuracy-compute reward:

```math
R = \mathbb{1}[\hat{y}=y] - \lambda \cdot \frac{\text{FLOPs}_i}{\text{FLOPs}_{full}}
```

This reward encourages the controller to find exits that are both correct and computationally cheap.

- Correct early exit → high reward
- Correct late exit → moderate reward
- Wrong early exit → low or negative reward
- Wrong late exit → worst reward

The hyperparameter `λ` controls how strongly computation is penalized.

### 9.3 Policy-Gradient Update

The controller can be trained using the REINFORCE update:

```math
\nabla J(\theta) = \mathbb{E}\left[R \nabla \log \pi_\theta(a_i \mid s_i)\right]
```

In implementation, this means:

1. Run an image through the early-exit ResNet.
2. At each exit, let the controller sample `exit` or `continue`.
3. When the controller exits, compute the final reward `R`.
4. Update the controller to increase the probability of actions that produced high reward.

A baseline value may be subtracted from the reward to reduce variance:

```math
\nabla J(\theta) = \mathbb{E}\left[(R - b) \nabla \log \pi_\theta(a_i \mid s_i)\right]
```

where `b` can be the moving average reward.

### 9.4 Why This Is Extra, Not Core

This phase is intentionally optional because policy-gradient training can be noisy and harder to debug than supervised controller training. The main project should first prove the early-exit system works through deterministic and easier-to-evaluate methods.

The REINFORCE phase will be considered successful if it can outperform or match the supervised learned controller under at least one compute-accuracy trade-off setting.

### 9.5 Extra Phase Evaluation

If implemented, this phase will add one more row to the final comparison table:

| Method | Accuracy | FLOPs Reduction | Latency Reduction | CO₂ Reduction | Avg Exit |
| :--- | ---: | ---: | ---: | ---: | ---: |
| REINFORCE exit controller | TBD | TBD | TBD | TBD | TBD |

The REINFORCE controller should be compared against:

- Fixed confidence thresholding
- Budget-aware dynamic thresholding
- Supervised learned exit controller
- Full ResNet-18 baseline

---

## 10. Immediate Next Step

The first concrete task is to train and evaluate the full ResNet-18 baseline on CIFAR-10. Do not start with the learned controller or REINFORCE phase before the baseline and fixed-threshold early-exit system work correctly.
