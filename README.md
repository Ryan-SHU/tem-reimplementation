# TEM Reimplementation

A PyTorch reimplementation of the Tolman-Eichenbaum Machine (TEM), based on:

> Whittington, J. C. R., Muller, T. H., Mark, S., Chen, G., Barry, C., Burgess, N., & Behrens, T. E. J. (2020).  
> **The Tolman-Eichenbaum Machine: Unifying Space and Relational Memory through Generalization in the Hippocampal Formation.**  
> *Cell*, 183(5), 1249–1263.e23.  
> https://doi.org/10.1016/j.cell.2020.10.024

This repository is a research-oriented implementation. The goal is not to exactly reproduce the original TensorFlow implementation line by line, but to build a readable, testable, and extensible PyTorch version of the model and its experimental pipeline.

---

## Project status

This project currently implements:

- Core TEM computation flow
- Multi-module grid/place latent states
- Attractor memory retrieval
- Generative and inference memory updates
- Full sequence forward pass
- TEM loss function
- Rectangular random-walk environment
- Random-walk batch generation
- Training loop
- Checkpointing
- CSV logging
- Basic representation analysis

Planned extensions:

- Hexagonal environments
- Family-tree relational environments
- Transitive inference environments
- More faithful reproduction of original experimental analyses
- Visualization of learned grid-like and place-like representations

---

## Repository structure

```text
tem_reimplementation/
├── configs/
│   └── tem_base.yaml
├── docs/
│   ├── computation_flow.md
│   └── data_generation_flow.md
├── scripts/
│   ├── train_rectangle.py
│   └── analyze_representations.py
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── environments.py
│   │   ├── walks.py
│   │   └── batches.py
│   ├── tem/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── losses.py
│   │   ├── types.py
│   │   ├── utils/
│   │   │   └── seed.py
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── activations.py
│   │       ├── tensor_ops.py
│   │       ├── attractor.py
│   │       └── tem.py
│   └── training/
│       ├── __init__.py
│       ├── logging.py
│       ├── checkpointing.py
│       └── trainer.py
└── tests/
    ├── test_activations.py
    ├── test_attractor.py
    ├── test_config.py
    ├── test_data.py
    ├── test_losses.py
    ├── test_tem_forward.py
    ├── test_tensor_ops.py
    └── test_training.py
```

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

In this implementation, the model receives only:

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

The model does not directly receive environment state ids or coordinates.

---

## Data pipeline

The experimental data pipeline is intentionally separated from the TEM model.

```text
src/data/environments.py
    Defines environment structure.

src/data/walks.py
    Samples random walks through the environment.

src/data/batches.py
    Converts integer walks into TEM-ready tensors.
```

The current implemented environment is:

```text
RectangleEnvironment
```

It defines a deterministic rectangular grid with four actions:

```text
0 = up
1 = down
2 = left
3 = right
```

The data pipeline produces batches with:

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

Only these fields are passed into the model:

```python
output = model(
    x=batch["x"],
    a=batch["a"],
    visited=batch["visited"],
)
```

The remaining fields are kept for analysis.

---

## Model computation

The full computation flow is documented in:

```text
docs/computation_flow.md
```

The implementation follows this sequence:

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

The central implementation is:

```text
src/tem/models/tem.py
```

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

Install dependencies:

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
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
pytest tests/test_data.py
pytest tests/test_training.py
```

---

## Training on a rectangle environment

Run a short debug training job:

```bash
python scripts/train_rectangle.py \
  --config configs/tem_base.yaml \
  --output-dir runs/rectangle_debug \
  --height 6 \
  --width 6 \
  --steps 100 \
  --log-every 10 \
  --checkpoint-every 50
```

Resume from the latest checkpoint:

```bash
python scripts/train_rectangle.py \
  --config configs/tem_base.yaml \
  --output-dir runs/rectangle_debug \
  --resume auto \
  --steps 100
```

Here, `--steps 100` means continuing for 100 additional optimization steps.

---

## Representation analysis

After training, run:

```bash
python scripts/analyze_representations.py \
  --config configs/tem_base.yaml \
  --checkpoint runs/rectangle_debug/checkpoints/latest.pt \
  --output-dir outputs/rectangle_debug_analysis \
  --height 6 \
  --width 6 \
  --num-batches 20
```

This computes average grid-state representations for each environment state.

Outputs:

```text
outputs/rectangle_debug_analysis/state_g_means.pt
outputs/rectangle_debug_analysis/state_g_means.csv
outputs/rectangle_debug_analysis/analysis_metadata.json
```

---

## Plotting results

After training and representation analysis, generate diagnostic plots with:

```bash
python scripts/plot_rectangle_results.py \
  --metrics-csv runs/rectangle_debug/metrics.csv \
  --state-g-csv outputs/rectangle_debug_analysis/state_g_means.csv \
  --output-dir outputs/rectangle_debug_plots \
  --height 6 \
  --width 6 \
  --top-k-units 6
```
This creates:
```
outputs/rectangle_debug_plots/
├── training_losses.png
├── latent_losses.png
├── state_visit_counts.png
├── module_0_rank_*_unit_*_rate_map.png
├── module_1_rank_*_unit_*_rate_map.png
└── plot_metadata.json

```
These plots are intended as early diagnostic figures. They should not be interpreted as a complete reproduction of the original paper's figures.

The current plots help answer:
- Does the training loss decrease?
- Are latent consistency losses finite and stable?
- Were all environment states sufficiently sampled?
- Do any grid-state units show spatially structured responses?


--

## Checkpoints and logs

Training outputs are written to:

```text
runs/
```

A typical run directory contains:

```text
runs/rectangle_debug/
├── metrics.csv
├── run_metadata.json
└── checkpoints/
    ├── latest.pt
    ├── step_0000050.pt
    └── step_0000100.pt
```

Generated outputs are intentionally ignored by Git.

---

## Recommended `.gitignore`

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

- readability
- explicit tensor shapes
- modular testing
- clear documentation
- separation between model, data, training, and analysis code

Because this is a reimplementation, numerical results may differ from the original paper unless the full experimental setup, hyperparameters, environments, and analysis procedures are matched carefully.


## Related repositories

The original authors provide TensorFlow implementations here:

- https://github.com/djcrw/generalising-structural-knowledge

A separate PyTorch implementation also exists:

- https://github.com/jbakermans/torch_tem

This repository is an independent educational and research-oriented PyTorch reimplementation.

---

## License

No license has been selected yet.

Before making this repository public, add a license file if you want others to be able to reuse, modify, or distribute the code.

Common choices include:

- MIT License
- Apache License 2.0
- BSD 3-Clause License

For academic research code, MIT or BSD 3-Clause are often simple choices.

---

## Disclaimer

This repository is for research and educational purposes.

It is not intended to be a drop-in reproduction of all experiments from the original paper at this stage.
