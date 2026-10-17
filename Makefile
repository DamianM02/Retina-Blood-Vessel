.PHONY: ruff-check ruff-format mypy test check pre-commit-install pre-commit-run download-dataset


ruff-check:
	uv run ruff check . # --fix .
ruff-format:
	uv run ruff format . # --check .
mypy:
	uv run mypy
test:
	uv run pytest
pre-commit-install:
	uv run pre-commit validate-config
	uv run pre-commit install

pre-commit-run:
	uv run pre-commit run --all-files

check: ruff-check mypy test

download-dataset:
	uv run src/vessel_seg/data/download.py
