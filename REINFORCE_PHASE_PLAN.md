# Policy-Gradient / REINFORCE Training Phase Plan

## 1. Objective

This phase trains a lightweight **exit controller** with Policy Gradient / REINFORCE to decide whether an input should:

- **exit** at the current intermediate classifier, or
- **continue** through deeper ResNet stages.

The objective is to maximize prediction quality while minimizing inference computation.

The central research question is:

> Can a REINFORCE-trained exit policy achieve a better accuracy–compute trade-off than fixed confidence thresholds, budget-aware dynamic thresholds, and a supervised reward-label controller?

---

## 2. Position in the Overall Project

This phase should begin only after the following components are complete:

1. Full ResNet-18 baseline trained on CIFAR-10.
2. Early-exit ResNet trained with auxiliary classifiers.
3. `forward_all_exits(images)` implemented and validated.
4. Per-exit accuracy and FLOPs measured.
5. Fixed confidence-threshold experiments completed.
6. Budget-aware dynamic-threshold experiments completed.
7. Preferably, a supervised reward-label controller implemented as a warm start.

Recommended project order:

```text
Phase 1: Full ResNet-18 baseline
Phase 2: Early-exit ResNet
Phase 3: Fixed confidence thresholding
Phase 4: Budget-aware dynamic thresholding
Phase 5: Supervised reward-label controller
Phase 6: REINFORCE controller training
```

The early-exit ResNet should be **frozen** during this phase. Only the controller parameters are updated.

---

## 3. System Formulation

### 3.1 Environment

A separate Gymnasium environment is not required.

The effective environment is:

```text
Frozen early-exit ResNet
+ one image and its label
+ sequential access to intermediate exit outputs
```

One image corresponds to one episode.

### 3.2 Agent

The agent is a small neural-network controller:

\[
\pi_\theta(a_i \mid s_i)
\]

where:

- \(	heta\) denotes controller parameters;
- \(s_i\) is the state observed at exit \(i\);
- \(a_i\) is the selected action.

### 3.3 Action space

At every non-final exit:

\[
a_i \in \{\text{continue}, \text{exit}\}
\]

Recommended encoding:

```text
0 = continue
1 = exit
```

At the final classifier, exit is mandatory.

### 3.4 Episode termination

An episode ends when:

- the controller selects `exit`; or
- the final classifier is reached.

The selected exit's prediction becomes the output for the image.

---

## 4. Required Model Interface

The frozen early-exit ResNet should expose all classifier outputs:

```python
logits_list = early_exit_model.forward_all_exits(images)
```

Expected structure:

```text
logits_list[0] -> Exit 1 logits, shape [B, C]
logits_list[1] -> Exit 2 logits, shape [B, C]
logits_list[2] -> Exit 3 logits, shape [B, C]
logits_list[3] -> Final-exit logits, shape [B, C]
```

Where:

- \(B\) is batch size;
- \(C\) is the number of classes.

Freeze the model before policy training:

```python
early_exit_model.eval()

for parameter in early_exit_model.parameters():
    parameter.requires_grad = False
```

The frozen model should remain in evaluation mode so that BatchNorm and Dropout behavior do not change during controller training.

---

## 5. State Representation

At exit \(i\), construct:

\[
s_i = [c_i, H_i, m_i, d_i, r_i]
\]

### 5.1 Maximum softmax confidence

\[
c_i = \max_j p_i(j)
\]

where:

\[
p_i = \operatorname{softmax}(z_i)
\]

and \(z_i\) is the exit's logit vector.

### 5.2 Predictive entropy

\[
H_i = -\sum_j p_i(j)\log p_i(j)
\]

Recommended normalization:

\[
H_i^{\text{norm}} = \frac{H_i}{\log C}
\]

This places entropy approximately in \([0,1]\).

### 5.3 Top-1/top-2 margin

\[
m_i = p_i(\text{top-1}) - p_i(\text{top-2})
\]

A larger margin normally indicates a clearer decision.

### 5.4 Normalized exit depth

\[
d_i = \frac{i}{K-1}
\]

where \(K\) is the total number of exits.

### 5.5 Normalized computation consumed

\[
r_i = \frac{\operatorname{FLOPs}_i}{\operatorname{FLOPs}_{\text{full}}}
\]

The per-exit FLOPs values must be measured before controller training and stored as constants.

### 5.6 State-building function

```python
def build_state(logits, exit_index, flops_ratio, num_exits):
    probabilities = torch.softmax(logits, dim=-1)

    confidence = probabilities.max(dim=-1).values

    entropy = -(
        probabilities
        * torch.log(probabilities.clamp_min(1e-8))
    ).sum(dim=-1)
    entropy = entropy / math.log(probabilities.size(-1))

    top_two = probabilities.topk(k=2, dim=-1).values
    margin = top_two[:, 0] - top_two[:, 1]

    normalized_depth = torch.full_like(
        confidence,
        exit_index / (num_exits - 1),
    )

    normalized_flops = torch.full_like(
        confidence,
        flops_ratio,
    )

    return torch.stack(
        [
            confidence,
            entropy,
            margin,
            normalized_depth,
            normalized_flops,
        ],
        dim=-1,
    )
```

---

## 6. Exit Controller Architecture

Use a small multilayer perceptron so that controller overhead remains negligible relative to the CNN.

```python
import torch
from torch import nn
from torch.distributions import Categorical


class ExitController(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, state):
        action_logits = self.network(state)
        return Categorical(logits=action_logits)
```

The categorical distribution provides:

```python
distribution = controller(state)

action = distribution.sample()
log_probability = distribution.log_prob(action)
policy_entropy = distribution.entropy()
```

During training, actions are sampled. During evaluation, actions should be deterministic.

---

## 7. Reward Design

### 7.1 Initial reward

Use:

\[
R = \mathbb{1}[\hat{y}=y] - \lambda r_i
\]

where:

- \(\hat{y}\) is the selected exit's prediction;
- \(y\) is the ground-truth label;
- \(r_i\) is normalized FLOPs;
- \(\lambda\) controls the compute penalty.

Interpretation:

- correct and early: high reward;
- correct and late: positive but smaller reward;
- wrong and early: negative compute penalty;
- wrong and late: larger negative compute penalty.

### 7.2 Alternative reward with explicit error penalty

If the controller tolerates too many wrong exits, use:

\[
R =
\begin{cases}
1 - \lambda r_i, & \hat{y}=y,\\
-\gamma - \lambda r_i, & \hat{y}\neq y.
\end{cases}
\]

where \(\gamma > 0\) penalizes incorrect predictions.

Recommended initial values:

```text
lambda_cost = 0.20
wrong_penalty gamma = 1.00, if needed
```

### 7.3 Reward sweep

Evaluate:

\[
\lambda \in \{0.05, 0.10, 0.20, 0.30, 0.50\}
\]

| Compute penalty | Expected policy |
|---:|---|
| 0.05 | Strongly accuracy-oriented |
| 0.10 | Mild pressure to exit early |
| 0.20 | Balanced starting point |
| 0.30 | Efficiency-oriented |
| 0.50 | Aggressive early exiting |

CodeCarbon should not be used directly inside the training reward. FLOPs provide a stable and inexpensive proxy. Carbon emissions should be measured during final evaluation.

---

## 8. REINFORCE Objective

For a trajectory containing actions from exits \(1\) through \(T\):

\[
\tau = (s_1,a_1,\ldots,s_T,a_T)
\]

the policy objective is:

\[
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R]
\]

The Monte Carlo policy-gradient estimate is:

\[
\nabla_\theta J(\theta)
\approx
(R-b)
\sum_{t=1}^{T}
\nabla_\theta \log\pi_\theta(a_t\mid s_t)
\]

where \(b\) is a baseline used to reduce variance.

The policy loss to minimize is:

\[
L_{\text{policy}}
=
-(R-b)
\sum_{t=1}^{T}
\log\pi_\theta(a_t\mid s_t)
\]

---

## 9. Variance Reduction

### 9.1 Moving-average baseline

Initialize:

```python
reward_baseline = 0.0
baseline_momentum = 0.9
```

Update after each batch:

```python
reward_baseline = (
    baseline_momentum * reward_baseline
    + (1.0 - baseline_momentum) * batch_mean_reward
)
```

Advantage:

\[
A = R-b
\]

The baseline should be detached and must not receive gradients.

### 9.2 Reward normalization

Optional improvement:

```python
advantages = rewards - rewards.mean()
advantages = advantages / (
    rewards.std(unbiased=False) + 1e-8
)
```

Try either:

- moving-average baseline; or
- batch reward normalization.

Do not introduce multiple variance-reduction changes simultaneously before establishing a baseline result.

---

## 10. Entropy Regularization

Add policy entropy to encourage exploration:

\[
L =
L_{\text{policy}}
-
\beta
\sum_{t=1}^{T}
\mathcal{H}(\pi_\theta(\cdot\mid s_t))
\]

Recommended values:

\[
\beta \in \{0.001,0.005,0.01\}
\]

Start with:

```text
entropy_beta = 0.005
```

Entropy regularization helps prevent:

- always exiting at Exit 1;
- always continuing to the final classifier;
- premature policy collapse.

It may be reduced gradually during training after the policy begins to stabilize.

---

## 11. Recommended Training Strategy

### 11.1 Preferred approach: supervised warm start

Before REINFORCE fine-tuning:

1. Run all exits for each image.
2. Compute the reward for every possible exit.
3. Select the exit with maximum reward.
4. Convert it to sequential labels:
   - continue before the selected exit;
   - exit at the selected exit.
5. Train the controller with cross-entropy loss.

Example:

| Exit | Correct | FLOPs ratio | Reward |
|---|---:|---:|---:|
| Exit 1 | No | 0.25 | -0.05 |
| Exit 2 | Yes | 0.50 | 0.90 |
| Exit 3 | Yes | 0.75 | 0.85 |
| Final | Yes | 1.00 | 0.80 |

For \(\lambda=0.2\), Exit 2 is the target:

```text
Exit 1 -> continue
Exit 2 -> exit
```

Then initialize the REINFORCE controller with these pretrained weights.

### 11.2 REINFORCE fine-tuning

After supervised pretraining:

1. Freeze the early-exit ResNet.
2. Sample actions from the controller.
3. Compute terminal reward.
4. Calculate REINFORCE loss.
5. Update only controller parameters.

This is expected to be more stable than starting REINFORCE from random controller weights.

---

## 12. Training Loop

### 12.1 Batch-level procedure

For every batch:

1. Compute all exit logits using the frozen ResNet under `torch.no_grad()`.
2. Process each sample as one episode.
3. At each non-final exit:
   - build the state;
   - obtain action distribution;
   - sample exit or continue;
   - store log-probability and entropy.
4. Stop at selected exit.
5. Compute prediction and terminal reward.
6. Calculate advantage.
7. Aggregate policy losses over the batch.
8. Backpropagate through the controller only.
9. Clip gradients.
10. Update moving-average baseline.

### 12.2 Reference pseudocode

```python
controller.train()
early_exit_model.eval()

reward_baseline = 0.0
baseline_momentum = 0.9

for epoch in range(num_epochs):
    for images, labels in training_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            exit_logits = early_exit_model.forward_all_exits(images)

        sample_losses = []
        sample_rewards = []

        for sample_index in range(images.size(0)):
            trajectory_log_probs = []
            trajectory_entropies = []

            selected_logits = None
            selected_flops_ratio = None

            for exit_index, batch_logits in enumerate(exit_logits):
                logits = batch_logits[
                    sample_index : sample_index + 1
                ]

                state = build_state(
                    logits=logits,
                    exit_index=exit_index,
                    flops_ratio=flops_ratios[exit_index],
                    num_exits=len(exit_logits),
                )

                is_final_exit = (
                    exit_index == len(exit_logits) - 1
                )

                if is_final_exit:
                    selected_logits = logits
                    selected_flops_ratio = flops_ratios[exit_index]
                    break

                distribution = controller(state)
                action = distribution.sample()

                trajectory_log_probs.append(
                    distribution.log_prob(action)
                )
                trajectory_entropies.append(
                    distribution.entropy()
                )

                if action.item() == 1:
                    selected_logits = logits
                    selected_flops_ratio = flops_ratios[exit_index]
                    break

            prediction = selected_logits.argmax(dim=-1)
            correct = (
                prediction
                == labels[sample_index : sample_index + 1]
            ).float()

            reward = (
                correct
                - lambda_cost * selected_flops_ratio
            ).squeeze(0)

            sample_rewards.append(reward)

            if trajectory_log_probs:
                log_prob_sum = torch.stack(
                    trajectory_log_probs
                ).sum()
                entropy_sum = torch.stack(
                    trajectory_entropies
                ).sum()
            else:
                log_prob_sum = torch.zeros((), device=device)
                entropy_sum = torch.zeros((), device=device)

            advantage = (
                reward - reward_baseline
            ).detach()

            sample_loss = (
                -advantage * log_prob_sum
                - entropy_beta * entropy_sum
            )

            sample_losses.append(sample_loss)

        batch_rewards = torch.stack(sample_rewards)
        batch_mean_reward = batch_rewards.mean().item()

        reward_baseline = (
            baseline_momentum * reward_baseline
            + (1.0 - baseline_momentum)
            * batch_mean_reward
        )

        policy_loss = torch.stack(sample_losses).mean()

        optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            controller.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
```

---

## 13. Hyperparameter Plan

Recommended starting configuration:

```yaml
controller:
  input_dim: 5
  hidden_dim: 32
  optimizer: adam
  learning_rate: 0.001
  epochs: 30
  batch_size: 128
  gradient_clip_norm: 1.0

reward:
  lambda_cost: 0.20
  wrong_penalty: 0.0

reinforce:
  entropy_beta: 0.005
  baseline_momentum: 0.90
  supervised_warm_start: true
```

Planned sweep:

| Parameter | Values |
|---|---|
| `lambda_cost` | 0.05, 0.10, 0.20, 0.30, 0.50 |
| `entropy_beta` | 0.001, 0.005, 0.010 |
| learning rate | \(10^{-4}\), \(5\times10^{-4}\), \(10^{-3}\) |
| hidden dimension | 16, 32, 64 |

Do not perform the full Cartesian product initially. First tune `lambda_cost`, then refine controller-training parameters.

---

## 14. Logging Requirements

Log the following for every epoch:

- mean training reward;
- validation reward;
- policy loss;
- policy entropy;
- controller action probabilities;
- validation accuracy;
- average FLOPs ratio;
- exit distribution;
- percentage of samples reaching final exit.

Suggested CSV schema:

```text
epoch,
train_reward,
validation_reward,
policy_loss,
policy_entropy,
validation_accuracy,
average_flops_ratio,
exit_1_rate,
exit_2_rate,
exit_3_rate,
final_exit_rate
```

Save:

```text
checkpoints/
  controller_best_reward.pt
  controller_best_accuracy_budget.pt
  controller_last.pt
```

The preferred checkpoint should be chosen on the validation set, not the test set.

---

## 15. Evaluation Protocol

### 15.1 Deterministic inference

Do not sample during final evaluation.

At each non-final exit:

```python
action = distribution.probs.argmax(dim=-1)
```

Equivalent decision:

```text
choose exit if P(exit) > P(continue)
```

The final exit remains mandatory.

### 15.2 Required metrics

Report:

- test accuracy;
- average FLOPs per sample;
- FLOPs reduction relative to full ResNet-18;
- mean and standard deviation of latency;
- CodeCarbon emissions;
- exit distribution;
- average selected exit;
- average reward.

### 15.3 Comparison methods

The final comparison must include:

1. Full ResNet-18.
2. Fixed confidence threshold.
3. Budget-aware dynamic threshold.
4. Supervised reward-label controller.
5. REINFORCE-trained controller.

Suggested table:

| Method | Accuracy | FLOPs Reduction | Latency Reduction | CO₂ Reduction | Average Exit |
|---|---:|---:|---:|---:|---:|
| Full ResNet-18 | TBD | 0% | 0% | 0% | Final |
| Fixed threshold | TBD | TBD | TBD | TBD | TBD |
| Dynamic threshold | TBD | TBD | TBD | TBD | TBD |
| Reward-label controller | TBD | TBD | TBD | TBD | TBD |
| REINFORCE controller | TBD | TBD | TBD | TBD | TBD |

### 15.4 Main plots

Generate:

1. Accuracy versus average FLOPs.
2. Accuracy versus latency.
3. Accuracy versus estimated emissions.
4. Reward versus training epoch.
5. Policy entropy versus training epoch.
6. Exit distribution by method.
7. Accuracy–compute frontier across \(\lambda\).

---

## 16. Ablation Studies

### 16.1 State-feature ablation

Compare:

```text
confidence only
confidence + entropy
confidence + entropy + margin
all features
```

### 16.2 Reward ablation

Compare:

- accuracy minus FLOPs penalty;
- explicit wrong-prediction penalty;
- different values of \(\lambda\).

### 16.3 Training ablation

Compare:

- REINFORCE from random initialization;
- supervised controller only;
- supervised warm start followed by REINFORCE.

### 16.4 Entropy ablation

Compare:

- no entropy regularization;
- fixed entropy coefficient;
- decaying entropy coefficient.

---

## 17. Common Failure Modes

### Policy always exits at Exit 1

Possible causes:

- compute penalty too large;
- wrong predictions are insufficiently penalized;
- entropy collapsed too early.

Actions:

- reduce \(\lambda\);
- introduce an explicit wrong-prediction penalty;
- increase the entropy coefficient;
- use supervised warm start.

### Policy always reaches final exit

Possible causes:

- compute penalty too small;
- initial policy strongly favors continuation;
- reward scale makes compute differences insignificant.

Actions:

- increase \(\lambda\);
- verify FLOPs normalization;
- initialize from reward-label supervision;
- train longer.

### Unstable reward

Possible causes:

- high REINFORCE variance;
- batch size too small;
- no baseline;
- learning rate too high.

Actions:

- use a moving-average baseline;
- normalize advantages;
- increase batch size;
- reduce learning rate;
- clip gradients.

### Good training reward but poor validation accuracy

Possible causes:

- controller overfitting;
- compute penalty too aggressive;
- deterministic evaluation differs from sampled training.

Actions:

- select checkpoints using validation metrics;
- reduce policy capacity;
- tune \(\lambda\);
- monitor both sampled and deterministic validation policies.

### Latency does not improve despite FLOPs reduction

Possible causes:

- all exits are computed before routing;
- routing occurs only after full computation;
- Python branching overhead dominates.

Actions:

- implement true conditional forward execution;
- benchmark per-sample and batch routing separately;
- distinguish theoretical FLOPs savings from measured latency savings;
- report both metrics honestly.

---

## 18. Implementation Checklist

### Model readiness

- [x] Early-exit ResNet checkpoint available.
- [x] All exits produce valid logits.
- [x] Final-exit accuracy validated.
- [x] Backbone frozen during policy training.
- [x] Per-exit FLOPs ratios measured.

### Controller

- [x] State builder implemented.
- [x] Controller MLP implemented.
- [x] Categorical action distribution implemented.
- [x] Final exit forced.
- [x] Deterministic evaluation mode implemented.

### Training

- [x] Reward function unit-tested.
- [x] Moving-average baseline implemented.
- [x] Entropy regularization implemented.
- [x] Gradient clipping implemented.
- [x] Supervised warm start available.
- [x] Checkpointing and logs implemented.

### Evaluation

- [ ] Accuracy measured.
- [ ] Average FLOPs measured.
- [ ] Exit distribution measured.
- [ ] Latency benchmarked.
- [ ] CodeCarbon run completed.
- [ ] Baselines evaluated under the same setup.

---

## 19. Acceptance Criteria

The phase is complete when:

1. The controller trains without numerical instability.
2. The policy does not collapse to one exit for all samples unless supported by the reward setting.
3. Deterministic validation metrics are reproducible.
4. At least three compute-penalty values are evaluated.
5. REINFORCE is compared fairly with fixed and dynamic thresholds.
6. Accuracy–FLOPs and accuracy–latency trade-offs are reported.
7. The selected policy provides either:
   - better accuracy at a comparable computation budget; or
   - lower computation at a comparable accuracy level.

REINFORCE does not need to dominate every baseline to produce a valid result. A negative result is still useful if the comparison is controlled and the causes are analyzed.

---

## 20. Expected Research Contribution

The REINFORCE phase extends the project from manually specified stopping criteria to a learned sequential policy.

The intended contribution is:

> A policy-gradient exit controller that directly optimizes the trade-off between classification correctness and inference computation, evaluated against fixed-threshold, dynamic-threshold, and supervised stopping policies.

The strongest practical implementation is:

```text
supervised reward-label pretraining
→ REINFORCE fine-tuning
→ deterministic budget-aware evaluation
```

---

## 21. Future Work

A later research direction is to transfer the same budget-aware stopping principle to AlphaZero-style policy-value MCTS.

Instead of choosing only an image-classification exit, a dynamic policy-value network could select shallow or deep evaluation paths under a fixed MCTS compute budget. This would study the trade-off between:

```text
deeper network + fewer MCTS simulations
versus
shallower network + more MCTS simulations
```

This remains future work and is outside the implementation scope of the current REINFORCE phase.
