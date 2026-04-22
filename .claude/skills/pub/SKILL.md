---
name: "Publication Skills"
description: "Research paper drafting, review, revision, and figure generation -- loaded when working on technical papers, white papers, or conference submissions"
domain: pub
type: skill
user-invocable: false
---

# Publications Domain

This project produces research papers in `paper/`. All publication work follows a structured draft-review-revise cycle with immutable version history and formal scoring, mirroring the IP domain's state machine.

## State Machine

```
EMPTY --> DRAFTED --> REVIEWED --> REVISED --> REVIEWED --> ... --> READY
```

Commands map to state transitions:

| Transition | Command | Input | Output |
|------------|---------|-------|--------|
| `EMPTY --> DRAFTED` | [[pub-draft]] | User interview + literature search | `{thread}.1/` |
| `DRAFTED --> REVIEWED` | [[pub-review]] | Version `{N}/` | `{N}.review/review.md` |
| `REVIEWED --> REVISED` | [[pub-revise]] | `{N}/` + `{N}.review/` | `{N+1}/` |
| `REVISED --> REVIEWED` | [[pub-review]] | Version `{N+1}/` | `{N+1}.review/review.md` |
| `READY --> AUDITED` | [[pub-audit]] | Version `{N}/` | `{N}.audit/audit.md` |
| any state | [[pub-figures]] | Paper `.tex` | Missing figure scripts in `figures/` |
| portfolio | [[pub]] | Thread name or all | Assessment + parallel agent launch |

Convergence criterion: review score >= 32/40 with 0 critical issues.

## Naming Convention

```
{thread}.{N}
```

- **thread**: the research directory name (e.g., `bispectral-layout-optimization`)
- **N**: integer starting at 1, incremented by `/pub-revise`

Example: `bispectral-layout-optimization.3` is the 3rd revision of the bispectral paper.

## Directory Layout

For this project, the thread name is `latent-symmetries`. Paper versions live in `paper/`:

```
paper/
  outline.md                     # Working outline (already exists)
  latent-symmetries.1/           # Draft version 1
    paper.tex                    # LaTeX paper (standard article class)
    paper.pdf                    # Compiled PDF
    literature.md                # Literature review / related work notes (internal)
    figures/                     # Python-generated figures (.py --> .pdf/.png)
    data/                        # Supporting scripts and results
  latent-symmetries.1.review/    # Review (read-only sibling)
    review.md                    # Scored review report (markdown)
  latent-symmetries.2/           # Revision incorporating review
    ...                          # Same structure
```

### Format Convention

- **LaTeX** (`.tex` --> `.pdf`): paper uses standard `\documentclass{article}` with academic packages (geometry, amsmath, booktabs, graphicx, hyperref). NOT `sphere-patent.sty`
- **Markdown** (`.md`): internal working documents -- `literature.md`, `review.md`
- **Python figures** (`.py` --> `.pdf` + `.png`): can use `scripts/gen_figures.py` for block diagrams, or raw matplotlib for data plots
- Compile: `pdflatex paper.tex` (run twice for refs)

### Key Principles

1. **Literature review first.** `/pub-draft` conducts a literature search before writing. Understand the landscape and position the contribution.
2. **Immutable versions.** Previous versions are never modified. The version history IS the revision trail.
3. **Separation of concerns.** Review is read-only (produces `{N}.review/`). Revision is separate (consumes `{N}/` + `{N}.review/`, produces `{N+1}/`).
4. **Converge, then submit.** Cycle until review score >= 32/40 with 0 critical issues.

### Existing Publications

Papers in progress should be listed in `README.md`.

## Commands

| Command | Description |
|---------|-------------|
| [[pub]] | Portfolio orchestrator: assess state of all papers, launch parallel agents |
| [[pub-draft]] | Interview + literature search + first-draft paper.tex (creates version 1) |
| [[pub-review]] | Read-only critic: 8-dimension scored review report |
| [[pub-revise]] | Consume draft + review, produce next version with all issues addressed |
| [[pub-audit]] | Fact-check: verify citations, numbers, equations, reproducibility claims |
| [[pub-figures]] | Batch-generate missing figures for a paper |

## Checkpointing Protocol

Long-running skills (`/pub-draft`, `/pub-revise`, `/pub-figures`) write `_progress.json` in the output directory. On retry, completed phases are skipped. Same `_progress.json` format and rules as the IP domain.

| Skill | Checkpointed Phases |
|-------|-------------------|
| `/pub-draft` | `interview`, `literature_search`, `paper_tex`, `compile_pdfs` |
| `/pub-revise` | `read_inputs`, `paper_tex`, `literature_update`, `compile_pdfs`, `self_check` |
| `/pub-figures` | `analysis`, then per-figure: `fig_N_script`, `fig_N_run`, `fig_N_verify` |
| `/pub-review` | Not checkpointed (single output file) |
| `/pub-audit` | Not checkpointed (single output file) |

## Key References

- `paper/outline.md` -- comprehensive paper outline with all experimental results
- `data/results/` -- JSON files with all experimental outputs
- `scripts/` -- experiment scripts (source of all reported results)

