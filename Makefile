# tensorspine — developer tasks. Run `make` (or `make help`) to list them.
# Assumes a POSIX-ish make (Linux/macOS/WSL/Git-Bash); the CI uses Ubuntu.

# Detect the virtualenv bin dir: Windows uses Scripts/, POSIX uses bin/.
ifeq ($(OS),Windows_NT)
    VENV_BIN := .venv/Scripts
else
    VENV_BIN := .venv/bin
endif
PY     := $(VENV_BIN)/python
PIP    := $(PY) -m pip
PYTEST := $(PY) -m pytest
RUFF   := $(PY) -m ruff

.DEFAULT_GOAL := help

.PHONY: help setup test test-fast gradcheck torch stubs progress \
        moons experiments lint fmt clean

help: ## List all targets with descriptions
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-13s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv, upgrade pip, install package editable with dev extras
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test: ## Run the full pytest suite (verbose)
	$(PYTEST) -v

test-fast: ## Run only the ops + accumulation tests (the fast iteration loop)
	$(PYTEST) -v tests/test_ops.py tests/test_accumulation.py

gradcheck: ## Run only the numerical gradient-checking tests
	$(PYTEST) -v tests/test_gradcheck.py

torch: ## Run only the PyTorch cross-check tests
	$(PYTEST) -v tests/test_vs_pytorch.py

stubs: ## List remaining NotImplementedError stubs and their count
	@python tools/stub_report.py

progress: ## Stub count plus a one-line test pass/fail summary
	@python tools/stub_report.py
	@echo ""
	@$(PYTEST) -q --no-header -p no:cacheprovider 2>&1 | tail -n 1

moons: ## Run the moons experiment (02) and save the decision-boundary plot
	$(PY) experiments/02_moons.py

experiments: ## Run every experiment script in order
	$(PY) experiments/01_single_neuron.py
	$(PY) experiments/02_moons.py
	$(PY) experiments/03_no_zero_grad.py
	$(PY) experiments/04_init_comparison.py
	$(PY) experiments/05_deep_chain.py

lint: ## Check style with ruff
	$(RUFF) check .

fmt: ## Auto-format with ruff
	$(RUFF) format .

clean: ## Remove caches and generated plot/graphviz artifacts
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -f experiments/*.png experiments/out/*.png experiments/output/*.png
	rm -f *.gv *.gv.pdf *.gv.png
