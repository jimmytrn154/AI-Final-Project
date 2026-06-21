# Figure Captions

## `accuracy_flops_frontier.png`

Accuracy versus FLOPs reduction for the fixed-threshold, dynamic-threshold, supervised-controller, and REINFORCE policy families. The fixed-threshold and REINFORCE sweeps occupy the strongest tradeoff region in the current CIFAR-10 experiments, while the supervised `best_reward` controller sacrifices substantial accuracy for more aggressive early exiting.

## `accuracy_vs_threshold.png`

Fixed-threshold ablation on CIFAR-10. Raising the confidence threshold steadily improves accuracy from 76.05% at `tau = 0.50` to 87.73% at `tau = 0.99`, while reducing the amount of computation saved from 71.68% to 32.83%.

## `controller_lambda_tradeoff.png`

Supervised `best_reward` controller sweep over `reward_lambda`. Changing the label-generation reward alters the controller's aggressiveness, but every setting in the current sweep remains far below the strongest threshold and REINFORCE baselines in accuracy.

## `reinforce_lambda_tradeoff.png`

REINFORCE controller sweep over `lambda_cost`. Lower cost weights preserve more accuracy, while larger cost weights encourage earlier exits and greater compute reduction at the expense of classification performance.

## `exit_distribution_comparison.png`

Exit-distribution comparison for representative methods. The dynamic accuracy-first policy collapses to the final head, the fixed-threshold policy spreads traffic across exits 2, 3, and the final head, the supervised controller exits overwhelmingly at the first head, and the strongest REINFORCE policy concentrates most samples at exits 2 and 3.

## `bootstrap_confidence_intervals.png`

Bootstrap 95% confidence intervals for representative method accuracies. The intervals come from deterministic prediction resampling rather than repeated training seeds, so they provide uncertainty estimates for the observed runs without implying formal statistical significance.

## `confusion_matrix_best_methods.png`

Confusion matrices for the baseline, early-exit final head, best fixed-threshold policy, best dynamic policy, best supervised controller, and best REINFORCE controller. The strongest threshold and REINFORCE policies remain visually close to the baseline, whereas the supervised controller shows much broader class confusion.
