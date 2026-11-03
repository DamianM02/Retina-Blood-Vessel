.PHONY: ruff-check ruff-format mypy test check pre-commit-install pre-commit-run after-pull download-dataset train


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

# Run after `git pull`: sync .venv with uv.lock and refresh pre-commit hooks
after-pull:
	uv sync --locked
	uv run pre-commit install
	uv run pre-commit install-hooks


download-dataset:
	uv run python -m vessel_seg.data.download

train:
	# -m cause module
	uv run python -m  vessel_seg.scripts.train
