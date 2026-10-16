# Retina Blood Vessel Segmentation

[![CI](https://github.com/DamianM02/Retina-Blood-Vessel/actions/workflows/ci.yml/badge.svg)](https://github.com/DamianM02/Retina-Blood-Vessel/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.14-blue)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

Semantic segmentation of blood vessels in retinal fundus images using U-Net,
built as a reproducible MLOps pipeline: from data versioning and experiment
tracking to model serving and monitoring.

> 🚧 **Work in progress.** The model was prototyped in Jupyter notebooks
> (`notebooks/`) and is being refactored into a tested Python package.

## Dataset

[Retina Blood Vessel](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel)
(Kaggle): 100 retinal images (80 train / 20 test) with binary vessel masks.

## Project structure

```
├── src/vessel_seg/       # Python package (data, models, training, inference)
├── tests/                # pytest test suite
├── notebooks/            # exploratory notebooks (original prototype)
├── .github/workflows/    # CI: lint, type check, tests
├── Makefile              # shortcuts for common dev commands
└── pyproject.toml        # dependencies and tool configuration
```

## Development setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:DamianM02/Retina-Blood-Vessel.git
cd Retina-Blood-Vessel
uv sync                  # create .venv and install dependencies
make pre-commit-install  # set up git hooks
make check               # lint + type check + tests
```

## Roadmap

- [x] Project setup: uv, ruff, mypy, pytest, pre-commit
- [x] CI with GitHub Actions
- [ ] Refactor notebook code into `src/vessel_seg` (dataset, model, training)
- [ ] Proper train / validation / test split
- [ ] Training with PyTorch Lightning + Hydra configs
- [ ] Experiment tracking & model registry (MLflow)
- [ ] Data & pipeline versioning (DVC)
- [ ] ONNX export + FastAPI inference service + Docker
- [ ] Interactive demo (Hugging Face Spaces)
- [ ] Monitoring
