# TEM Reimplementation

A PyTorch reimplementation framework for the Tolman-Eichenbaum Machine (TEM), based on:

> Whittington, J. C. R., Muller, T. H., Mark, S., Chen, G., Barry, C., Burgess, N., & Behrens, T. E. J. (2020).  
> **The Tolman-Eichenbaum Machine: Unifying Space and Relational Memory through Generalization in the Hippocampal Formation.**  
> *Cell*, 183(5), 1249–1263.e23.  
> https://doi.org/10.1016/j.cell.2020.10.024

This repository is an independent research-oriented PyTorch implementation. The goal is to build a readable, testable, and extensible codebase for reproducing the major TEM experiments, not to copy the original TensorFlow implementation line by line.

---

## Current status

This repository currently contains the core TEM model and a clean experiment framework.

| Component | Status |
|---|---|
| Core TEM forward computation | Implemented |
| Multi-module grid/place latent states | Implemented |
| Attractor memory retrieval | Implemented |
| Generative and inference memory updates | Implemented |
| Full sequence forward pass | Implemented |
| TEM loss function | Implemented |
| YAML configuration loading | Implemented |
| Generic training utilities | Implemented |
| Checkpointing utilities | Implemented |
| CSV logging utilities | Implemented |
| Generic data interface | Scaffolded |
| Experiment 1: structural generalization | Scaffolded |
| Experiment 2: spatial representations | Scaffolded |
| Experiment 3: transition-statistics cell types | Scaffolded |
| Experiment 4: remapping analysis | Scaffolded |

This repository should currently be understood as a **clean TEM experiment framework**, not yet a complete reproduction of all paper figures.

---

## Reproduction roadmap

The full project is organized around four experiment groups from Whittington et al. (2020).

| Experiment group | Paper-level question | Planned repository area |
|---|---|---|
| Experiment 1: structural generalization | Can TEM generalize structural knowledge to novel sensory environments? | `exp1_generalization` |
| Experiment 2: spatial representations | Does TEM learn grid-like and place-like representations in spatial graphs? | `exp2_spatial` |
| Experiment 3: transition statistics | Can different transition statistics produce border, object-vector, landmark, and task-structure cells? | `exp3_transition_statistics` |
| Experiment 4: remapping analysis | Is structural knowledge preserved across hippocampal remapping? | `exp4_remapping` |

The current scripts are dry-run experiment entry points. They define the expected command-line interface and output structure before the full experiment implementations are added.

---

## Repository structure

```text
tem_reimplementation/
├── LICENSE
├── README.md
├── configs/
│   ├── experiments/
│   └── tem_base.yaml
├── docs/
│   ├── computation_flow.md
│   └── experiments/
├── projectLog.md
├── pyproject.toml
├── scripts/
│   ├── exp1_generalization/
│   ├── exp2_spatial/
│   ├── exp3_transition_statistics/
│   └── exp4_remapping/
├── src/
│   ├── tem/
│   │   ├── config.py
│   │   ├── losses.py
│   │   ├── types.py
│   │   ├── models/
│   │   │   ├── activations.py
│   │   │   ├── attractor.py
│   │   │   ├── tem.py
│   │   │   └── tensor_ops.py
│   │   └── utils/
│   │       └── seed.py
│   ├── tem_analysis/
│   ├── tem_data/
│   │   ├── base.py
│   │   ├── envs/
│   │   ├── queries/
│   │   └── sampling/
│   ├── tem_experiments/
│   │   ├── exp1_generalization/
│   │   ├── exp2_spatial/
│   │   ├── exp3_transition_statistics/
│   │   └── exp4_remapping/
│   └── tem_training/
│       ├── checkpointing.py
│       ├── logging.py
│       ├── trainer.py
│       └── types.py
└── tests/
    ├── test_activations.py
    ├── test_attractor.py
    ├── test_config.py
    ├── test_losses.py
    ├── test_tem_forward.py
    ├── test_tensor_ops.py
    └── test_training.py
```

Generated files such as `__pycache__/`, `.pytest_cache/`, `runs/`, `outputs/`, checkpoints, and `.pt` files should not be committed.

---

## Conceptual overview

TEM is a model of structural and relational memory. It combines:

1. **Path integration**
   - Uses actions or relations to update an abstract grid-like latent state.

2. **Sensory inference**
   - Uses current observations to infer latent state through memory.

3. **Attractor memory**
   - Retrieves bound place-like representations from learned associative memories.

4. **Generative prediction**
   - Predicts sensory observations from latent memory retrieval.

5. **Hebbian-style memory updates**
   - Updates generative and inference memories during experience.

In this implementation, the model receives:

```text
x:       [B, T, N_x]
a:       [B, T, N_a]
visited: [B, T]
```

where:

```text
B   = batch size
T   = sequence length
N_x = number of sensory observation categories
N_a = number of action / relation categories
```

The model does not directly receive environment state ids, coordinates, graph nodes, task labels, or query labels. Those variables belong to data generation, evaluation, and analysis code.

---

## Core model

The central model implementation is:

```text
src/tem/models/tem.py
```

The main loss implementation is:

```text
src/tem/losses.py
```

The full computation flow is documented in:

```text
docs/computation_flow.md
```

At a high level, the computation is:

```text
x_t
    -> sensory compression
    -> module-specific sensory trace
    -> p-space sensory cue

g_{t-1}, a_t
    -> transition-based grid prediction

sensory cue, inference memory
    -> memory-based grid estimate

path estimate + memory estimate
    -> inferred grid state

grid state + sensory cue
    -> bound place representation

grid state + generative memory
    -> sensory prediction

place representation + retrieved memory
    -> Hebbian memory update
```

---

## Code organization

The project separates model code, data code, training utilities, analysis utilities, and experiment-specific logic.

```text
src/tem/
    Core TEM model, losses, configuration, model outputs, and tensor utilities.

src/tem_data/
    Generic data interfaces, environment definitions, samplers, and relational query utilities.

src/tem_training/
    Generic trainer, checkpointing, logging, and batch-provider types.

src/tem_analysis/
    Reusable analysis utilities.

src/tem_experiments/
    Reusable experiment-specific logic.

scripts/
    Command-line entry points for running each experiment group.
```

The intended dependency direction is:

```text
scripts/
    -> tem_experiments/
        -> tem_data/
        -> tem_analysis/
        -> tem_training/
        -> tem/
```

The `tem` package should not depend on experiment-specific code.

---

## Installation

Clone the repository:

```bash
git clone git@github.com:Ryan-SHU/tem-reimplementation.git
cd tem-reimplementation
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
pip install -e .
```

Install development dependencies if needed:

```bash
pip install pytest matplotlib tqdm pyyaml
```

Install PyTorch according to your platform and CUDA version.

---

## Running tests

Run all tests:

```bash
pytest
```

Run selected tests:

```bash
pytest tests/test_tem_forward.py
pytest tests/test_losses.py
pytest tests/test_training.py
```

---

## Experiment scripts

Each experiment group has a command-line entry point.

At the current stage, these scripts support `--dry-run`. A dry run writes an `experiment_plan.json` file describing the intended experiment, expected outputs, and missing implementation components.

### Experiment 1: structural generalization

```bash
python scripts/exp1_generalization/run.py \
  --config configs/tem_base.yaml \
  --output-dir outputs/exp1_generalization_dry_run \
  --dry-run
```

Planned targets:

```text
transitive inference
family tree relational inference
2D graph link inference
zero-shot structural generalization
```

---

### Experiment 2: spatial representations

```bash
python scripts/exp2_spatial/run.py \
  --config configs/tem_base.yaml \
  --output-dir outputs/exp2_spatial_dry_run \
  --dry-run
```

Planned targets:

```text
2D spatial graph training
grid-like g-cell analysis
place-like p-cell analysis
rate maps
gridness scores
cross-environment remapping analysis
```

---

### Experiment 3: transition-statistics cell types

```bash
python scripts/exp3_transition_statistics/run.py \
  --config configs/tem_base.yaml \
  --output-dir outputs/exp3_transition_statistics_dry_run \
  --dry-run
```

Planned targets:

```text
border cells
object-vector cells
landmark cells
lap/task-structure cells
transition-statistics-dependent representations
```

---

### Experiment 4: remapping analysis

```bash
python scripts/exp4_remapping/run.py \
  --config configs/tem_base.yaml \
  --output-dir outputs/exp4_remapping_dry_run \
  --dry-run
```

Planned targets:

```text
simulated remapping analysis
place-grid relationship preservation
cross-environment structural relationship analysis
optional real neural dataset analysis
```

---

## Experiment outputs

Generated experiment outputs should be written to:

```text
outputs/
```

Training runs and checkpoints should be written to:

```text
runs/
```

Both directories are ignored by Git.

---

## Development workflow

Recommended branch naming:

```text
main
    Stable branch. Should pass tests.

refactor/*
    Architecture changes.

exp1/*
    Structural generalization experiments.

exp2/*
    Spatial representation experiments.

exp3/*
    Transition-statistics experiments.

exp4/*
    Remapping experiments.

fix/*
    Bug fixes.

docs/*
    Documentation-only changes.
```

Recommended workflow:

```bash
git checkout main
git pull
git checkout -b exp1/structural-generalization
```

After changes:

```bash
pytest
git status
git add .
git commit -m "Add structural generalization scaffold"
git push -u origin exp1/structural-generalization
```

Then open a pull request into `main`.

---

## Git hygiene

Before committing, check for generated files:

```bash
git ls-files | grep -E '(__pycache__|\.pyc$|runs/|outputs/|checkpoints/|\.pt$|\.pth$|\.egg-info)' || true
```

Remove Python caches locally:

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
```

Recommended `.gitignore` entries:

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Virtual environments
.venv/
venv/
env/

# Build artifacts
build/
dist/
*.egg-info/

# Experiment outputs
runs/
outputs/
checkpoints/
*.pt
*.pth

# OS/editor files
.DS_Store
.vscode/
.idea/
```

---

## Citation

If this repository is useful for your work, please cite the original TEM paper:

```bibtex
@article{whittington2020tolman,
  title = {The Tolman-Eichenbaum Machine: Unifying Space and Relational Memory through Generalization in the Hippocampal Formation},
  author = {Whittington, James C. R. and Muller, Timothy H. and Mark, Shirley and Chen, Guifen and Barry, Caswell and Burgess, Neil and Behrens, Timothy E. J.},
  journal = {Cell},
  volume = {183},
  number = {5},
  pages = {1249--1263.e23},
  year = {2020},
  doi = {10.1016/j.cell.2020.10.024}
}
```

---

## Relationship to the original work

This repository is an independent PyTorch reimplementation for study and experimentation.

It is based on the published TEM model and is not an official release from the original authors.

The implementation emphasizes:

- readability;
- explicit tensor shapes;
- modular testing;
- clear documentation;
- separation between model, data, training, analysis, and experiment code.

Because this is a reimplementation, numerical results may differ from the original paper unless the full experimental setup, hyperparameters, environments, and analysis procedures are carefully matched.

---

## Related repositories

The original authors provide TensorFlow implementations here:

- https://github.com/djcrw/generalising-structural-knowledge

A separate PyTorch implementation also exists:

- https://github.com/jbakermans/torch_tem

This repository is an independent educational and research-oriented PyTorch reimplementation.

---

## License

This project is released under the MIT License.

See:

```text
LICENSE
```

---

## Disclaimer

This repository is for research and educational purposes.

It is not intended to be a drop-in reproduction of all experiments from the original paper at this stage.
