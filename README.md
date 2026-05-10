# TEM Reimplementation

A simple PyTorch implementation of the Tolman-Eichenbaum Machine.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
```

## Test
```bash
pytest
```

## Main structure
```
configs/
  tem_base.yaml

docs/
  computation_flow.md

src/tem/
  config.py
  types.py
  models/
  data/
  training/
  utils/

tests/
  test_config.py

```

## Computation flow

See:
```
docs/computation_flow.md
```
