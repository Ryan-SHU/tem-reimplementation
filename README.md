# TEM Reimplementation

A research-oriented PyTorch reimplementation of the Tolman-Eichenbaum Machine (TEM), based on:

> Whittington, J. C. R., Muller, T. H., Mark, S., Chen, G., Barry, C., Burgess, N., & Behrens, T. E. J. (2020).  
> **The Tolman-Eichenbaum Machine: Unifying Space and Relational Memory through Generalization in the Hippocampal Formation.**  
> *Cell*, 183(5), 1249–1263.e23.  
> https://doi.org/10.1016/j.cell.2020.10.024

This repository is a research-oriented PyTorch implementation of TEM. The goal is not to copy the original TensorFlow implementation line by line, but to build a readable, testable, and extensible implementation that can eventually reproduce the major experimental results from the paper.

---

## Project status

This repository is under active development.

Current implementation status:

| Component | Status |
|---|---|
| Core TEM forward computation | Implemented |
| Multi-module grid/place latent states | Implemented |
| Attractor memory retrieval | Implemented |
| Generative and inference memory updates | Implemented |
| Full sequence forward pass | Implemented |
| TEM loss function | Implemented |
| YAML configuration loading | Implemented |
| Training loop | Implemented |
| Checkpointing | Implemented |
| CSV logging | Implemented |
| Rectangle random-walk environment | Implemented |
| Random-walk batch generation | Implemented |
| Basic rectangle representation analysis | Implemented |
| Basic rectangle diagnostic plotting | Implemented |
| Experiment 1: structural generalization | Planned |
| Experiment 2: spatial representations | Partially implemented |
| Experiment 3: transition-statistics cell types | Planned |
| Experiment 4: remapping analysis | Planned |

This repository should currently be understood as a **TEM research framework in progress**, not yet a complete reproduction of all results or figures from Whittington et al. (2020).

---

## Reproduction roadmap

The planned full reproduction is organized around four experiment groups from the paper.

| Experiment group | Paper-level question | Repository area | Current status |
|---|---|---|---|
| Experiment 1 | Can TEM generalize structural knowledge to novel sensory environments? | `exp1_generalization` | Planned |
| Experiment 2 | Does TEM learn grid-like and place-like representations in spatial graphs? | `exp2_spatial` | Partially implemented |
| Experiment 3 | Can transition statistics produce border, object-vector, landmark, and task-structure cells? | `exp3_transition_statistics` | Planned |
| Experiment 4 | Is structural knowledge preserved across hippocampal remapping? | `exp4_remapping` | Planned |

The current rectangle experiment belongs to **Experiment 2** and is intended as an early diagnostic pipeline.

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
│   ├── data_generation_flow.md
│   └── experiments/
├── scripts/
│   └── exp2_spatial/
│       ├── train_rectangle.py
│       ├── analyze_rectangle_representations.py
│       └── plot_rectangle_results.py
├── src/
│   ├── tem/
│   │   ├── config.py
│   │   ├── losses.py
│   │   ├── types.py
│   │   ├── utils/
│   │   └── models/
│   │       ├── activations.py
│   │       ├── attractor.py
│   │       ├── tensor_ops.py
│   │       └── tem.py
│   ├── tem_data/
│   │   ├── base.py
│   │   ├── environments.py
│   │   ├── walks.py
│   │   ├── batches.py
│   │   ├── envs/
│   │   │   ├── base.py
│   │   │   ├── grid.py
│   │   │   └── rectangle.py
│   │   ├── queries/
│   │   └── sampling/
│   ├── tem_training/
│   │   ├── checkpointing.py
│   │   ├── logging.py
│   │   └── trainer.py
│   ├── tem_analysis/
│   └── tem_experiments/
│       ├── exp1_generalization/
│       ├── exp2_spatial/
│       ├── exp3_transition_statistics/
│       └── exp4_remapping/
└── tests/
```

The core principle is:

```text
src/tem/
    Model, loss, config, and tensor-level TEM computation.

src/tem_data/
    Data interfaces, environments, sampling logic, and query generation.

src/tem_training/
    Generic training utilities independent of any specific experiment.

src/tem_analysis/
    Shared analysis utilities such as rate maps, gridness, remapping metrics,
    and plotting helpers.

src/tem_experiments/
    Reusable experiment-level Python logic.

scripts/
    Command-line entry points for running specific experiments.
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
   - Retrieves bound place-like representations from associative memories.

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
N_a = number of action or relation categories
```

The model does **not** directly receive environment state ids, coordinates, graph nodes, or task labels. Those variables are used only by data generation, evaluation, and analysis code.

---

## Model computation

The full computation flow is documented in:

```text
docs/computation_flow.md
```

At a high level, the implementation follows this sequence:

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
    -> memory update
```

The central model implementation is:

```text
src/tem/models/tem.py
```

The main loss implementation is:

```text
src/tem/losses.py
```

---

## Data and experiment organization

The project separates model code from experiment code.

```text
src/tem/
    Core model, losses, configuration, and model-specific types.

src/tem_data/
    Generic environment interfaces, graph environments, random walks,
    batch generation, sampling utilities, and relational query utilities.

src/tem_training/
    Generic training loop, checkpointing, and logging.

src/tem_analysis/
    Reusable analysis utilities.

src/tem_experiments/
    Reusable experiment-specific logic.

scripts/
    Command-line entry points for running experiments.
```

The current implemented data pipeline supports a rectangle random-walk task:

```text
RectangleEnvironment
    -> random walk sampler
    -> batch generator
    -> TEM model
    -> loss
    -> trainer
    -> checkpoint and metrics
    -> representation analysis
    -> diagnostic plots
```

The current batch format is:

```python
batch = {
    "x": x,
    "a": a,
    "visited": visited,
    "position": position,
    "observation_id": observation_id,
    "action_id": action_id,
}
```

Only these fields are model inputs:

```python
output = model(
    x=batch["x"],
    a=batch["a"],
    visited=batch["visited"],
)
```

The remaining fields are kept for analysis.

---

## Data and experiment design

The cleaned repository intentionally separates generic data interfaces from experiment-specific environments.

The base environment interface is:

```text
src/tem_data/base.py
```

A discrete TEM environment should define:

```text
states:
    hidden graph nodes or task states

actions:
    movements, relations, or task transitions

transition:
    state, action -> next state

observation:
    state -> sensory observation id
```

The TEM model itself only sees one-hot observations and one-hot actions or relations. Experiment-specific variables such as node ids, positions, object identities, lap indices, or family-tree relations should be kept outside the model and used only for data generation and analysis.

---

## Training interface

The generic trainer depends only on a minimal batch-provider interface.

A batch provider must implement:

```python
sample_batch() -> dict
```

and return at least:

```python
{
    "x": x,              # [B, T, N_x]
    "a": a,              # [B, T, N_a]
    "visited": visited,  # [B, T]
}
```

Additional fields may be included for analysis, for example:

```python
{
    "state": state,
    "position": position,
    "observation_id": observation_id,
    "action_id": action_id,
    "query": query,
}
```

The trainer does not need to know what these experiment-specific fields mean.

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

Install this package in editable mode:

```bash
pip install -e .
```

Install development dependencies if needed:

```bash
pip install pytest matplotlib tqdm pyyaml
```

Install PyTorch according to your platform and CUDA version. For example, if you need CUDA support, install the PyTorch build that matches your local CUDA version before running experiments.

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

## Experiment 2: rectangle spatial diagnostic

The current executable experiment is a rectangle random-walk diagnostic task under:

```text
scripts/exp2_spatial/
```

This is an early version of the spatial representation experiment. It is useful for checking whether:

- the model trains without numerical failure;
- prediction losses decrease;
- latent states remain finite;
- states are sufficiently sampled;
- some latent units show spatial modulation.

It should not yet be interpreted as a full reproduction of the original paper's spatial representation figures.

---

### Train rectangle model

Run a short debug training job:

```bash
python scripts/exp2_spatial/train_rectangle.py \
  --config configs/tem_base.yaml \
  --output-dir runs/rectangle_debug \
  --height 6 \
  --width 6 \
  --steps 100 \
  --log-every 10 \
  --checkpoint-every 50
```

Run a longer diagnostic job:

```bash
python scripts/exp2_spatial/train_rectangle.py \
  --config configs/tem_base.yaml \
  --output-dir runs/rectangle_long \
  --height 6 \
  --width 6 \
  --steps 5000 \
  --log-every 100 \
  --checkpoint-every 1000
```

Resume from the latest checkpoint:

```bash
python scripts/exp2_spatial/train_rectangle.py \
  --config configs/tem_base.yaml \
  --output-dir runs/rectangle_long \
  --resume auto \
  --steps 1000
```

Here, `--steps 1000` means continuing for 1000 additional optimization steps.

---

### Analyze rectangle representations

After training, collect state-wise mean grid representations:

```bash
python scripts/exp2_spatial/analyze_rectangle_representations.py \
  --config configs/tem_base.yaml \
  --checkpoint runs/rectangle_long/checkpoints/latest.pt \
  --output-dir outputs/rectangle_long_analysis \
  --height 6 \
  --width 6 \
  --num-batches 200
```

This produces:

```text
outputs/rectangle_long_analysis/
├── state_g_means.pt
├── state_g_means.csv
└── analysis_metadata.json
```

The analysis computes:

```text
mean_g[state, unit] = average g-unit activity over visits to that state
```

At the current cleaned stage, these directories are placeholders. They are intentionally kept separate so that each paper experiment can be implemented, tested, and reviewed independently.

### Plot rectangle diagnostics

Generate diagnostic plots:

```bash
python scripts/exp2_spatial/plot_rectangle_results.py \
  --metrics-csv runs/rectangle_long/metrics.csv \
  --state-g-csv outputs/rectangle_long_analysis/state_g_means.csv \
  --output-dir outputs/rectangle_long_plots \
  --height 6 \
  --width 6 \
  --top-k-units 6
```

This creates:

```text
outputs/rectangle_long_plots/
├── training_losses.png
├── latent_losses.png
├── state_visit_counts.png
├── module_*_rank_*_unit_*_rate_map.png
├── summary_figure.png
├── summary_figure.pdf
└── plot_metadata.json
```

The combined summary figure includes:

```text
A. Observation prediction losses
B. Latent losses and gradient norm
C. State visit counts
D/E/... Most spatially varying g-units per module
```

These plots are diagnostic figures. They should not be interpreted as a formal reproduction of the original paper's figures.

---

## Experiment outputs

Training outputs are written to:

```text
runs/
```

Analysis and plotting outputs are written to:

```text
outputs/
```

A typical training run directory looks like:

```text
runs/rectangle_long/
├── metrics.csv
├── run_metadata.json
└── checkpoints/
    ├── latest.pt
    ├── step_0001000.pt
    ├── step_0002000.pt
    └── ...
```

These generated files are intentionally ignored by Git.

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
git checkout -b exp2/gridness-analysis
```

After making changes:

```bash
pytest
git status
git add .
git commit -m "Add gridness analysis utilities"
git push -u origin exp2/gridness-analysis
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

The recommended Git workflow is:

```text
main
    Stable branch. Should pass tests.

refactor/*
    Structural changes to the codebase.

cleanup/*
    Removal of obsolete files, generated outputs, or unused code.

exp1/*
    Structural generalization experiments.

exp2/*
    Spatial representation experiments.

exp3/*
    Transition-statistics experiments.

exp4/*
    Remapping experiments.

docs/*
    Documentation-only updates.

fix/*
    Bug fixes.
```

Recommended development sequence:

```bash
git checkout main
git pull
git checkout -b exp1/structural-generalization
```

After making changes:

```bash
pytest
git status
git add .
git commit -m "Implement initial structural generalization environment"
git push -u origin exp1/structural-generalization
```

Then open a pull request into `main`.

---

## Versioning plan

Suggested milestones:

```text
v0.1-core-tem
    Core TEM model, loss, config, and tests.

v0.2-training-framework
    Generic trainer, checkpointing, logging, and batch-provider interface.

v0.3-experiment-framework
    Clean data, analysis, and experiment namespaces.

v0.4-exp1-generalization
    Structural generalization experiments.

v0.5-exp2-spatial
    Spatial representation and gridness experiments.

v0.6-exp3-transition-statistics
    Border, object-vector, landmark, and task-structure experiments.

v0.7-exp4-remapping
    Simulated and/or neural-data remapping analysis.
```

Create a tag after stable milestones:

```bash
git tag -a v0.3-experiment-framework -m "Clean experiment framework"
git push origin v0.3-experiment-framework
```

---

## Generated files and Git hygiene

Generated experiment outputs should not be committed.

The repository should ignore:

```text
runs/
outputs/
checkpoints/
*.pt
*.pth
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/
```

Before committing, it is useful to check:

```bash
git status
git ls-files | grep -E '(__pycache__|\.pyc$|runs/|outputs/|checkpoints/|\.pt$|\.pth$|\.egg-info)' || true
```

If generated files appear in Git tracking, remove them from the index:

```bash
git rm -r --cached runs outputs checkpoints
git ls-files | grep -E '(__pycache__|\.pyc$|\.pt$|\.pth$|\.egg-info)' | xargs -r git rm -r --cached
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

```text
https://github.com/djcrw/generalising-structural-knowledge
```

A separate PyTorch implementation also exists:

```text
https://github.com/jbakermans/torch_tem
```

This repository is an independent PyTorch reimplementation for study, testing, and experiment reproduction.

---

## Relationship to the original work

This repository is based on the published TEM model but is not an official release from the original authors.

The implementation emphasizes:

- readable PyTorch code
- explicit tensor shapes
- modular testing
- separation between model, data, training, analysis, and experiment code
- reproducible experiment scripts
- careful distinction between implemented results and planned reproduction work

Because this is a reimplementation, numerical results may differ from the original paper unless the full experimental setup, hyperparameters, environments, training protocol, and analysis procedures are matched carefully.

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

It is not currently a complete reproduction of all experiments or figures from Whittington et al. (2020). Experiment-specific code and analysis will be added incrementally.
