# Conference / Project Guidelines

## Source

These guidelines are adapted from the COMP2050 Artificial Intelligence Programming Project description dated May 3, 2026.

## 1. Project Goal

The project must study one AI algorithm in detail.

The project should include:

- One AI algorithm of the student's choice.
- One or more student-proposed variations of the algorithm.
- An experimental study of the impact of those variations.
- A research-style report documenting the method, experiments, results, and findings.

The selected algorithm does not need to be limited to algorithms covered in class.

The proposed variations do not need to outperform the original algorithm. The evaluation should focus on the quality of the work, the clarity of the study, implementation effort, and experimental analysis, not only on whether the variation improves the numerical result.

## 2. Environment and Tools

The project must use one or more environments, datasets, or problem settings to evaluate the original algorithm and its variations.

The student may use:

- Existing AI environments or frameworks.
- A custom-built environment.
- Any programming language suitable for the project.

Suggested environment examples from the assignment include:

- Gymnasium
- OpenSpiel
- Neural MMO
- MAgent 2
- Fighting ICE
- Legend of Code and Magic
- AdLeap-MAS

These are suggestions only; other frameworks, datasets, or environments are allowed.

For this project, the selected environment/dataset is CIFAR-10, and the chosen AI topic is adaptive inference / early-exit image classification.

## 3. Report Format Requirements

The final report must follow the overall structure of a research paper.

There is no strict template or page limit, but the report should contain the following sections:

1. Title and student name
2. Abstract
3. Introduction
4. Related Work
5. Methodology
6. Results
7. Discussion
8. Conclusion
9. Acknowledgements
10. References

The abstract should be short, preferably one paragraph.

The report may be written in LaTeX. LaTeX is recommended but not required.

## 4. Figures and Tables

The report should present results using research-paper-style figures and tables.

Requirements:

- Every figure must have a number and caption.
- Every table must have a number and caption.
- Figures should be referenced in the text.
- Tables should be referenced in the text.
- Plots should be clear, readable, and relevant to the experimental analysis.

Recommended tools for plots include Matplotlib or R, but other tools are acceptable.

For this project, recommended figures include:

- Accuracy comparison across methods.
- Accuracy versus FLOPs.
- Accuracy versus latency.
- CO2 or energy-efficiency comparison.
- Ablation results for thresholds or controller reward parameters.

## 5. Experimental Evaluation Requirements

The report must include a detailed experimental analysis.

The evaluation may study the impact of:

- Student-proposed variations of the original algorithm.
- Hyperparameters.
- Problem settings.
- Different environments or datasets.
- Efficiency and performance tradeoffs.

For this project, the evaluation should focus on:

- Baseline classifier performance.
- Early-exit classifier performance.
- Fixed-threshold exit policy.
- Dynamic-threshold exit policy.
- Learned controller policy.
- REINFORCE-based controller policy.
- Accuracy, latency, FLOPs, and estimated CO2/energy tradeoffs.
- Threshold ablations.
- Controller lambda/reward ablations.

## 6. Statistical Analysis Requirements

The report is strongly recommended to include statistical analysis.

At minimum, the report should consider multiple runs so that mean and variance can be studied.

If multiple runs are not possible, the report must justify why.

For this project, the report should include one of the following:

- Multiple-seed results with mean and variance.
- Repeated experimental runs.
- Ablation tables showing the stability of conclusions across settings.
- A clear justification if only one run is available.

Do not claim statistical significance unless the analysis supports it.
nowledge them in the Acknowledgements section.

## 11. Grading Criteria

The project is graded according to the following distribution:

| Criterion | Weight |
|---|---:|
| Quality of writing and presentation | 20% |
| Creativity | 20% |
| Implementation quality and effort | 30% |
| Experimental evaluation | 30% |

The report should therefore prioritize:

- Clear research-paper-style writing.
- Creative algorithmic variation or experimental framing.
- Strong implementation evidence.
- Detailed and well-presented experimental evaluation.

## 14. PaperOrchestra-Specific Instructions

When using PaperOrchestra or Codex to generate the report:

1. Use this file as the project guideline file.
2. Use `experimental_log.md` as the source of truth for experimental results.
3. Use `idea.md` as the source of truth for the research idea and contribution framing.
4. Use metrics files, CSV files, JSON files, and run manifests as evidence.
5. Do not invent numbers or experimental outcomes.
6. Do not invent citations.
7. Preserve research-paper structure.
8. Include a clear Acknowledgements section.
9. Include a clear limitation discussion.
10. Make sure all figures and tables have numbers and captions.
11. Make sure final outputs are suitable for `report.pdf`.

## 15. Project-Specific Checklist

Before finalizing the report, verify that:

- [ ] The report has a title and student name.
- [ ] The abstract is one paragraph.
- [ ] The report includes Introduction, Related Work, Methodology, Results, Discussion, Conclusion, Acknowledgements, and References.
- [ ] The methodology explains the baseline and all proposed variations.
- [ ] The results include plots and tables.
- [ ] Every figure has a number and caption.
- [ ] Every table has a number and caption.
- [ ] Experimental results include mean/variance or justify why multiple runs were not possible.
- [ ] The report discusses the quality and limitations of the results.

## 16. Deadline and Literature Cutoff

The target project deadline is July 12, 2026.

For the literature review, prioritize foundational and recent work relevant to efficient inference, early-exit neural networks, dynamic neural networks, adaptive computation, confidence-based exiting, reinforcement learning for sequential decisions, and Green AI.

The literature review should prefer papers available before the final report deadline. Do not invent citations. If a citation cannot be verified, mark it as `TODO: citation needed`.