PYTHON ?= python
PKG    := rl_final

.PHONY: install format check test clean \
        train-baseline sweep-entropy sweep-count sweep-rnd \
        plots reproduce pre-commit

# --- Setup -------------------------------------------------------------------

install:
	uv pip install -e ".[dev]"
	pre-commit install

# --- Code quality ------------------------------------------------------------

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

check:
	ruff format --check src tests scripts
	ruff check src tests scripts

pre-commit:
	pre-commit run --all-files

# --- Tests -------------------------------------------------------------------

test:
	pytest tests/ -v

# --- Individual runs (for debugging) -----------------------------------------

train-baseline:
	$(PYTHON) -m $(PKG).ppo.train env=empty_8x8 bonus=none seed=0

# --- Sweeps (main experiments) -----------------------------------------------

sweep-entropy:
	$(PYTHON) -m $(PKG).ppo.train --multirun \
	  env=empty_8x8,doorkey_8x8,multiroom \
	  bonus=high_entropy \
	  bonus.c_ent=0.05,0.1,0.2 \
	  seed=0,1,2,3,4

sweep-count:
	$(PYTHON) -m $(PKG).ppo.train --multirun \
	  env=empty_8x8,doorkey_8x8,multiroom \
	  bonus=count \
	  bonus.beta=0.01,0.1,1.0 \
	  seed=0,1,2,3,4

sweep-rnd:
	$(PYTHON) -m $(PKG).ppo.train --multirun \
	  env=empty_8x8,doorkey_8x8,multiroom \
	  bonus=rnd \
	  bonus.beta=0.01,0.1,1.0 \
	  seed=0,1,2,3,4

# --- Analysis ----------------------------------------------------------------

plots:
	$(PYTHON) scripts/make_plots.py

# One command reproduces the whole study. Use for the final submission.
reproduce: sweep-entropy sweep-count sweep-rnd plots

# --- Cleanup -----------------------------------------------------------------

clean:
	rm -rf data/runs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
