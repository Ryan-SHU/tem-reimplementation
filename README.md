# TEM Reimplementation

A research-oriented PyTorch reimplementation of the Tolman-Eichenbaum Machine (TEM), based on:

> Whittington, J. C. R., Muller, T. H., Mark, S., Chen, G., Barry, C., Burgess, N., & Behrens, T. E. J. (2020).  
> **The Tolman-Eichenbaum Machine: Unifying Space and Relational Memory through Generalization in the Hippocampal Formation.**  
> *Cell*, 183(5), 1249–1263.e23.  
> https://doi.org/10.1016/j.cell.2020.10.024

This repository is an independent educational and research implementation. The goal is to build a readable, testable, and extensible PyTorch codebase for studying the TEM model and eventually reproducing the major experiment groups from the original paper.

This is **not** an official implementation from the original authors.

---

## Project status

This repository is under active development.

The current codebase contains the cleaned core implementation and experiment framework scaffold. Earlier rectangle-debug experiment scripts and generated outputs have been removed so that the repository can be extended cleanly toward the full paper reproduction.

Current implementation status:

| Component | Status |
|---|---|
| Core TEM forward computation | Implemented |
| Multi-module grid/place latent states | Implemented |
| Attractor memory retrieval | Implemented |
| Generative and inference memory updates | Implemented |
| TEM sequence loss | Implemented |
| Configuration loading | Implemented |
| Training loop | Implemented |
| Checkpointing | Implemented |
| CSV logging | Implemented |
| Generic batch-provider interface | Implemented |
| Generic discrete-environment interface | Implemented |
| Experiment-specific data pipelines | Planned |
| Experiment-specific training scripts | Planned |
| Experiment-specific analysis scripts | Planned |
| Full reproduction of paper experiments | Planned |

At this stage, the repository should be understood as a **clean TEM model and experiment-framework base**, not yet as a complete reproduction of all paper figures.

---

## Reproduction roadmap

The full reproduction is organized around four experiment groups from Whittington et al. (2020).

| Experiment group | Paper result | Repository namespace | Current status |
|---|---|---|---|
| Experiment 1 | Structural generalization across new sensory environments | `exp1_generalization` | Planned |
| Experiment 2 | Grid-like and place-like representations in spatial graphs | `exp2_spatial` | Planned |
| Experiment 3 | Border cells, object-vector cells, landmark cells, and task-structure cells | `exp3_transition_statistics` | Planned |
| Experiment 4 | Structural preservation across hippocampal remapping | `exp4_remapping` | Planned |

The intended development order is:

```text
1. Build clean experiment framework
2. Implement Experiment 1: structural generalization
3. Implement Experiment 2: spatial representations and gridness analysis
4. Implement Experiment 3: transition-statistics cell types
5. Implement Experiment 4: remapping analysis
```

---

## Repository structure

```text
tem_reimplementation/
├── LICENSE
├── README.md
├── configs/
│   ├── tem_base.yaml
│   └── experiments/
├── docs/
│   ├── computation_flow.md
│   └── experiments/
├── scripts/
│   ├── exp1_generalization/
│   ├── exp2_spatial/
│   ├── exp3_transition_statistics/
│   └── exp4_remapping/
├── src/
│   ├── tem/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── losses.py
│   │   ├── types.py
│   │   ├── utils/
│   │   └── models/
│   │       ├── activations.py
│   │       ├── attractor.py
│   │       ├── tem.py
│   │       └── tensor_ops.py
│   ├── tem_data/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── envs/
│   │   ├── sampling/
│   │   └── queries/
│   ├── tem_training/
│   │   ├── __init__.py
│   │   ├── checkpointing.py
│   │   ├── logging.py
│   │   ├── trainer.py
│   │   └── types.py
│   ├── tem_analysis/
│   │   └── __init__.py
│   └── tem_experiments/
│       ├── __init__.py
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

In this implementation, the model receives tensors of the form:

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

The model does not directly receive environment state ids, graph nodes, coordinates, or task labels. Those variables belong to the data-generation and analysis layers.

---

## Model computation

The full computation flow is documented in:

```text
docs/computation_flow.md
```

At a high level, the sequence update is:

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

The central implementation is:

```text
src/tem/models/tem.py
```

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

Install PyTorch for your platform first. For example, follow the official PyTorch installation selector for your operating system, Python version, and CUDA version.

Then install this repository in editable mode:

```bash
pip install -e .
```

For development, make sure the test and utility dependencies are available:

```bash
pip install pytest tqdm pyyaml matplotlib
```

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

The current tests focus on the core TEM model, tensor operations, losses, attractor memory, configuration logic, and generic training loop.

---

## Running experiments

Experiment-specific scripts will be added under:

```text
scripts/exp1_generalization/
scripts/exp2_spatial/
scripts/exp3_transition_statistics/
scripts/exp4_remapping/
```

At the current cleaned stage, these directories are placeholders. They are intentionally kept separate so that each paper experiment can be implemented, tested, and reviewed independently.

Future examples will follow the pattern:

```bash
python scripts/exp1_generalization/<script_name>.py
python scripts/exp2_spatial/<script_name>.py
python scripts/exp3_transition_statistics/<script_name>.py
python scripts/exp4_remapping/<script_name>.py
```

---

## Development workflow

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
