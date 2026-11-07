.PHONY: check lint test env smoke smoke-mock hub-init

# Offline-safe: lint plus tests. The tiny ungated model is fetched once and cached.
check: lint test

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

env:
	uv run oluso env

# P0 gate. Needs the Tiny Aya Earth licence accepted and HF_TOKEN set.
smoke:
	uv run oluso smoke --backend hf

smoke-mock:
	uv run oluso smoke --backend mock

hub-init:
	uv run oluso hub-init --apply
