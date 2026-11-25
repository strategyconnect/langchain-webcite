.PHONY: install check lint typecheck test build

install:
	uv sync --all-groups

lint:
	uv run ruff check langchain_webcite tests examples
	uv run ruff format --check langchain_webcite tests examples

typecheck:
	uv run mypy

test:
	uv run pytest -q

check: lint typecheck test

build:
	rm -rf dist && uv build
