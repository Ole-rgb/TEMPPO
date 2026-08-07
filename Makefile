PYTHON ?= python
PKG    := rl_final

.PHONY: install format check test clean \
        sanity run-ppo run-rnd run-rnd-llm sweep-beta-llm \
        plots reproduce pre-commit

TRAIN := $(PYTHON) -m $(PKG).ppo.ppo_minigrid
ENV   ?= doorkey_8x8
SEEDS := 0,21,42

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

# --- Sanity check (build step 1) ---------------------------------------------

sanity:
	$(TRAIN) --multirun env=empty_5x5 bonus=none seed=$(SEEDS) total_timesteps=60000

# --- The four conditions (build step 4) --------------------------------------
# One codebase, flip coefficients. Same env-step budget for all four.

run-ppo:                     ## beta_rnd=0, beta_llm=0
	$(TRAIN) --multirun env=$(ENV) bonus=none    seed=$(SEEDS)

run-rnd:                     ## BASELINE
	$(TRAIN) --multirun env=$(ENV) bonus=rnd     seed=$(SEEDS)

run-rnd-llm:                 ## PROPOSED
	$(TRAIN) --multirun env=$(ENV) bonus=rnd_llm seed=$(SEEDS)

# The headline result: beta_rnd held fixed, beta_llm swept.
sweep-beta-llm:
	$(TRAIN) --multirun env=$(ENV) bonus=rnd_llm seed=$(SEEDS) llm.beta=0.1,0.25,0.5,1.0

# --- Analysis ----------------------------------------------------------------

plots:
	$(PYTHON) -m $(PKG).analysis.plots

# One command reproduces the whole study. Use for the final submission.
reproduce: sweep-beta-llm run-ppo run-rnd run-rnd-llm plots

# --- Cleanup -----------------------------------------------------------------

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
