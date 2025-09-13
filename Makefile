.PHONY: help install install-dev clean lint format test run-train-incremental run-train-evaluator run-train-random run-manual-exp run-comet-grids run-diff-corrector shell check-deps

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
install-poetry: ## Install Poetry package manager
	curl -sSL https://install.python-poetry.org | python3 -
	@echo "Poetry installed! Add /home/ubuntu/.local/bin to your PATH:"
	@echo "export PATH=\"/home/ubuntu/.local/bin:\$$PATH\""

setup-path: ## Add Poetry to PATH in current session
	export PATH="/home/ubuntu/.local/bin:$$PATH"

verify-poetry: ## Verify Poetry installation
	/home/ubuntu/.local/bin/poetry --version

install: ## Install dependencies using Poetry
	/home/ubuntu/.local/bin/poetry install

install-dev: ## Install dependencies including dev dependencies
	/home/ubuntu/.local/bin/poetry install --with dev

clean: ## Clean Poetry cache and virtual environment
	/home/ubuntu/.local/bin/poetry cache clear --all pypi
	rm -rf .venv

# Code quality
lint: ## Run linting with ruff
	/home/ubuntu/.local/bin/poetry run ruff check .

format: ## Format code with ruff
	/home/ubuntu/.local/bin/poetry run ruff format .

check-deps: ## Check for outdated dependencies
	/home/ubuntu/.local/bin/poetry show --outdated

# Development
shell: ## Activate Poetry shell
	/home/ubuntu/.local/bin/poetry shell

# Training scripts
run-train-incremental: ## Run incremental training
	/home/ubuntu/.local/bin/poetry run python -m src.pipelines.training.train_incremental

run-train-evaluator: ## Run training evaluator
	/home/ubuntu/.local/bin/poetry run python -m src.pipelines.training.train_evaluator

run-train-random: ## Run random regressor training
	/home/ubuntu/.local/bin/poetry run python -m src.pipelines.training.train_random_regressor

# Scripts
run-manual-exp: ## Create manual experiment
	/home/ubuntu/.local/bin/poetry run python scripts/create_manual_experiment.py

run-comet-grids: ## Run Comet ML grids
	/home/ubuntu/.local/bin/poetry run python scripts/make_comet_grids.py

run-diff-corrector: ## Run diff corrector evaluation
	/home/ubuntu/.local/bin/poetry run python scripts/run_diff_corrector_evaluation.py

run-diff-corrector-plotter: ## Run diff corrector plotter
	/home/ubuntu/.local/bin/poetry run python scripts/run_diff_corrector_plotter.py

# Streamlit dashboard
dashboard: ## Start Streamlit dashboard
	/home/ubuntu/.local/bin/poetry run streamlit run dashboard/med_wav.py

# Pre-commit hooks
pre-commit-install: ## Install pre-commit hooks
	/home/ubuntu/.local/bin/poetry run pre-commit install

pre-commit-run: ## Run pre-commit on all files
	/home/ubuntu/.local/bin/poetry run pre-commit run --all-files

# Testing
test: ## Run tests
	/home/ubuntu/.local/bin/poetry run python -m pytest tests/

# Project info
info: ## Show project information
	@echo "Project: Med-WAV"
	@echo "Python version: $(shell /home/ubuntu/.local/bin/poetry run python --version)"
	@echo "Poetry version: $(shell /home/ubuntu/.local/bin/poetry --version)"
	@echo "Virtual environment: $(shell /home/ubuntu/.local/bin/poetry env info --path)"

# Quick setup for new developers
setup: install-dev pre-commit-install ## Complete setup for new developers
	@echo "Setup complete! Run 'make help' to see available commands."

setup-full: install-poetry setup-path install-dev pre-commit-install ## Complete setup including Poetry installation
	@echo "Full setup complete! Poetry installed and dependencies ready."
	@echo "Run 'make help' to see available commands."
