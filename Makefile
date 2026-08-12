PYTHON ?= python
PKG    := rl_final

TRAIN  := $(PYTHON) -m $(PKG).ppo.ppo_minigrid
SWEEPS := multirun

.PHONY: install format check pre-commit test sanity audit warm-cache \
        run-ppo run-rnd run-llm run-rnd-llm run-warmstart run-no-goal \
        plots tensorboard reproduce clean

# --- Experimental constants ---------------------------------------------------
ENV       ?= multiroom_n4s5
SEEDS     ?= 0 21 42 63 84 105
TIMESTEPS ?= 750_000
JOBS      ?= 6

PIN := llm.skip_go_to_goal=false

empty     :=
space     := $(empty) $(empty)
comma     := ,
SEEDS_CSV := $(subst $(space),$(comma),$(SEEDS))
PLOT_ENVS := $(subst $(comma),$(space),$(ENV))

define sweep
	@mkdir -p logs
	@echo "  $(1) on $(ENV): $(words $(SEEDS)) seeds, $(JOBS) at a time -> logs/$(1)-$(ENV)-seed*.log"
	@echo "$(SEEDS)" | tr ' ' '\n' | xargs -P $(JOBS) -I{} sh -c \
	  'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(TRAIN) --multirun \
	     hydra.sweep.dir=$(SWEEPS)/$(1)/$(ENV)/seed$$1 \
	     env=$(ENV) total_timesteps=$(TIMESTEPS) seed=$$1 $(PIN) $(2) \
	     > logs/$(1)-$(ENV)-seed$$1.log 2>&1' sh {}
	@echo "  done -> $(SWEEPS)/$(1)/$(ENV)/"
endef

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

test:
	pytest tests/ -v

sanity:
	$(TRAIN) env=empty_5x5 bonus=none seed=0 total_timesteps=50_000

# --- Prerequisites for any LLM condition -------------------------------------
# Run these IN ORDER before the first llm/rnd_llm sweep on a new environment.

# Is env cacheable at all? `plan_classes` must stay bounded and `coverage` near
# 1.0, or the cache never closes and most episodes train with no bonus (KeyCorridor:
# 5,260 classes, 13% coverage -- unusable). Costs no API budget; run it first.
AUDIT_ENV ?= MiniGrid-MultiRoom-N4-S5-v1
audit:
	$(PYTHON) -c "from $(PKG).bonus.llm import audit_env; print(audit_env('$(AUDIT_ENV)'))"


# Gets one plan per plan class via the LLM and appends it to llm.cache_path.
warm-cache:
	$(TRAIN) --multirun env=$(ENV) bonus=rnd_llm seed=0 \
	  total_timesteps=20_000 llm.cache_only=false $(PIN)


# --- Conditions ---------------------------------------------------------------
run-ppo:
	$(call sweep,ppo,bonus=none)

run-rnd:                     ## BASELINE
	$(call sweep,rnd,bonus=rnd)

run-llm:                     ## does the LLM add anything beyond novelty?
	$(call sweep,llm,bonus=llm)

run-rnd-llm:                 ## constant beta_llm
	$(call sweep,rnd_llm,bonus=rnd_llm)

# Warm start: beta_llm decays linearly to 0 over anneal_llm_steps
run-warmstart:
	$(call sweep,warmstart_150k,bonus=rnd_llm_warmstart bonus.anneal_llm_steps=150000)
	$(call sweep,warmstart_300k,bonus=rnd_llm_warmstart bonus.anneal_llm_steps=300000)

# The one condition that deliberately inverts $(PIN): drops go_to_goal from every
# bound plan.
run-no-goal: PIN := llm.skip_go_to_goal=true
run-no-goal:
	$(call sweep,rnd_llm_no_goal,bonus=rnd_llm)

# --- Analysis -----------------------------------------------------------------

# Reads every sweep under $(SWEEPS)/. build_score_tensors intersects eval-step grids
# across ALL runs, so one unfinished sweep truncates every curve to wherever it
# stopped -- check the printed "eval points: N, up to X env steps" line before
# trusting a figure, and move aborted sweeps out of $(SWEEPS)/ rather than leaving them.
plots:
	$(PYTHON) -m $(PKG).analysis.plots --run-dir $(SWEEPS) \
	  --env $(PLOT_ENVS) --reps 10_000

tensorboard:
	tensorboard --logdir $(SWEEPS)

# --- Full study ---------------------------------------------------------------
reproduce: audit warm-cache run-ppo run-rnd run-llm run-rnd-llm run-warmstart run-no-goal plots

# --- Cleanup -------------------------------------------------------------------
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
