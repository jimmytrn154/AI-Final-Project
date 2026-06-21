# AGENTS.md

## Project Overview

This repository contains the COMP2050 Artificial Intelligence Programming Project:

**Learning When to Stop: Budget-Aware Early Exiting for Energy-Efficient CNN Inference**

The project studies adaptive inference for image classification using a ResNet-18 backbone on CIFAR-10. The main idea is to reduce unnecessary computation by allowing easy images to exit early while harder images continue deeper into the network.

The project compares:

* Full ResNet-18 baseline
* Early-exit ResNet-18
* Fixed confidence thresholding
* Budget-aware dynamic thresholding
* Supervised learned exit controller
* REINFORCE-trained exit controller

## Repository Layout

Main project code:

* `src/ai_final_project/` — package source code
* `scripts/` — training, evaluation, ablation, plotting, and analysis scripts
* `configs/` — experiment configuration files
* `tests/` — unit and output tests
* `checkpoints/` — model checkpoints; do not modify unless explicitly asked
* `data/` — CIFAR-10 data; do not modify unless explicitly asked

Experiment outputs:

* `outputs/metrics/` — root-level result CSV/JSON files
* `outputs/plots/` — existing generated plots
* `outputs/runs/2026-06-20-deeper-experiments/` — main final experiment run
* `outputs/runs/2026-06-19-direct-latency/` — earlier supporting experiment run

PaperOrchestra workspace:

* `workspace/inputs/idea.md` — project framing and contributions
* `workspace/inputs/experimental_log.md` — experiment evidence rules and metric priority
* `workspace/inputs/conference_guidelines.md` — report requirements
* `workspace/inputs/template.tex` — LaTeX report template
* `workspace/inputs/figures/` — existing figures for audit/reuse
* `workspace/figures/` — final audited or regenerated figures
* `workspace/final/` — final paper outputs

PaperOrchestra skills:

* `.agents/skills/paper-orchestra/`
* `.agents/skills/outline-agent/`
* `.agents/skills/plotting-agent/`
* `.agents/skills/literature-review-agent/`
* `.agents/skills/section-writing-agent/`
* `.agents/skills/content-refinement-agent/`

## Main Goal for Paper-Writing Tasks

When asked to run PaperOrchestra or write the report, use the skills in `.agents/skills` and the workspace in `./workspace`.

Expected final outputs:

* `workspace/final/paper.tex`
* `workspace/final/paper.pdf`, if LaTeX is available
* `workspace/figures/figure_manifest.md`
* `workspace/figures/figure_captions.md`
* `workspace/figures/plotting_notes.md`

## Source-of-Truth Rules

Use:

* `workspace/inputs/idea.md` for project framing, research question, hypothesis, and contributions.
* `workspace/inputs/experimental_log.md` for experiment design, metric priority order, allowed claims, and output-folder evidence rules.
* `workspace/inputs/conference_guidelines.md` for report structure and grading requirements.
* `workspace/inputs/template.tex` as the LaTeX structure to fill.

Use numeric evidence from output files only.

Priority order for numeric evidence:

1. `outputs/runs/2026-06-20-deeper-experiments/metrics/`
2. `outputs/runs/2026-06-20-deeper-experiments/run_manifest.json`
3. `outputs/metrics/`
4. `PROGRESS.md`

If the same metric appears in multiple files and values conflict, use the higher-priority source and record the conflict in notes.

## Strict Research Integrity Rules

Do not invent:

* datasets
* experiments
* metrics
* numeric results
* baselines
* ablations
* citations
* claims of improvement
* claims of statistical significance

If a value is missing, write:

```text
TODO: verify
```

or:

```text
N/A
```

If a citation cannot be verified, write:

```text
TODO: citation needed
```

Clearly distinguish:

* directly measured values
* estimated or baseline-scaled values
* unsupported hypotheses
* verified conclusions

Do not claim statistical significance unless repeated runs, variance analysis, or robustness analysis supports it.

## Code Safety Rules

Do not modify training code unless explicitly asked.

Do not rerun heavy ML experiments unless explicitly asked.

Do not overwrite:

* `outputs/metrics/`
* `outputs/runs/`
* `checkpoints/`
* `data/`

When generating paper assets, write to:

* `workspace/figures/`
* `workspace/drafts/`
* `workspace/final/`
* `workspace/refinement/`

Do not write generated paper artifacts into `src/`, `scripts/`, `configs/`, `outputs/`, or `checkpoints/` unless explicitly asked.

## Figure and Plotting Rules

Before generating new figures, audit existing figures in:

* `workspace/inputs/figures/`
* `outputs/plots/`

Reuse existing figures if they are:

* accurate
* readable
* consistent with source metrics
* suitable for a two-column LaTeX report
* supported by captions and evidence

Regenerate or add figures only if:

* a figure is inconsistent with metrics
* a figure is unreadable
* a figure lacks needed labels or units
* a required analysis is missing
* the figure is not suitable for the report format

All plotted numbers must come from CSV/JSON metrics files or values explicitly recorded in `experimental_log.md`.

For every final figure, produce or update:

* `workspace/figures/figure_manifest.md`
* `workspace/figures/figure_captions.md`
* `workspace/figures/plotting_notes.md`

Every figure must have:

* a number
* a caption
* a text reference in the paper
* a documented data source

## Paper Structure Requirements

The final report should follow a research-paper style structure:

* Title and student name
* Abstract
* Introduction
* Related Work
* Methodology
* Experimental Setup
* Results
* Discussion
* Limitations
* Conclusion
* Acknowledgements
* References

Figures and tables must be numbered and captioned.

The report should be suitable for a compact double-column, IEEE-like format.

## Validation Commands

Before running the full PaperOrchestra pipeline, run:

```bash
python external/PaperOrchestra/skills/paper-orchestra/scripts/validate_inputs.py --workspace workspace
```

If `.agents/skills/paper-orchestra/` is available and working, this is also acceptable:

```bash
python .agents/skills/paper-orchestra/scripts/validate_inputs.py --workspace workspace
```

Before finalizing, check for unresolved TODOs:

```bash
grep -R "TODO" workspace/final workspace/figures || true
```

If LaTeX is available, compile the final paper from `workspace/final/`.

## Testing Commands

For code-level checks, use:

```bash
pytest
```

Do not run long training scripts unless explicitly requested.

## Completion Criteria

A PaperOrchestra report-generation task is complete when:

* `workspace/final/paper.tex` exists
* `workspace/final/paper.pdf` exists if LaTeX is available
* every numeric result in the paper is supported by a metrics file
* every figure and table has a caption and text reference
* unsupported claims are marked as `TODO: verify`
* unverified citations are marked as `TODO: citation needed`
* final notes summarize which files were created and what TODOs remain
