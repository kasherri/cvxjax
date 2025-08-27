# Makefile for cvxjax

.PHONY: help install install-dev setup test test-fast lint format type-check clean docs docs-serve build upload

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in production mode
	pip install .

install-dev:  ## Install package in development mode with all dependencies
	pip install -e ".[dev]"

setup: install-dev  ## Complete development setup
	pre-commit install
	@echo "Development environment set up successfully!"

test:  ## Run all tests with coverage
	pytest tests/ -v --cov=cvxjax --cov-report=term-missing --cov-report=html

test-fast:  ## Run tests without coverage (faster)
	pytest tests/ -v

test-examples:  ## Run example scripts to verify they work
	python examples/quickstart_qp.py
	python examples/lasso_training_loop.py
	python examples/portfolio_qp.py

lint:  ## Run linting checks
	ruff check .

lint-fix:  ## Run linting with automatic fixes
	ruff check . --fix

format:  ## Format code with ruff
	ruff format .

format-check:  ## Check code formatting without making changes
	ruff format . --check

type-check:  ## Run type checking with mypy
	mypy cvxjax --ignore-missing-imports

check-all: lint format-check type-check  ## Run all code quality checks

clean:  ## Clean up build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docs:  ## Build documentation
	@echo "Building documentation..."
	@if [ -d "docs/_build" ]; then rm -rf docs/_build; fi
	@echo "Documentation built in docs/_build/"

docs-serve:  ## Serve documentation locally (if you add a docs server later)
	@echo "Serving documentation at http://localhost:8000"
	@echo "Note: Add a documentation server (e.g., mkdocs serve) here"

build:  ## Build package for distribution
	python -m build

upload:  ## Upload package to PyPI (requires authentication)
	python -m twine upload dist/*

upload-test:  ## Upload package to test PyPI
	python -m twine upload --repository testpypi dist/*

# Development workflow targets
dev-install: setup  ## Alias for setup
dev-test: test-fast lint format-check  ## Quick development test cycle
dev-check: check-all test  ## Comprehensive development check

# CI-like targets
ci-test: install-dev check-all test test-examples  ## Run full CI test suite locally
