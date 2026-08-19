PYTHON ?= python
PKG    := rl_final

TRAIN  := $(PYTHON) -m $(PKG).ppo
SWEEPS := multirun

.PHONY: help install format check pre-commit test sanity audit warm-cache \
        run-ppo run-rnd run-llm run-rnd-llm run-warmstart run-shuffled run-entcoef \
        run-lockedroom sensitivity conditions \
        sweep-beta-llm sweep-beta-rnd \
        plots tensorboard reproduce hpo hpo-incumbents clean

# --- Experimental constants ---------------------------------------------------
ENV       ?= multiroom_n4s5
SEEDS     ?= 0 21 42 63 84 105 126 147 168 189
TIMESTEPS ?= 750_000
JOBS      ?= 6

# Pinned on every sweep even though both match the config defaults
PIN := llm.skip_go_to_goal=false llm.cache_only=true

# Extra Hydra overrides appended to every run of a sweep, e.g.
# EXTRA=bonus.anneal_llm_frac=0.15.
EXTRA ?=

empty     :=
space     := $(empty) $(empty)
comma     := ,
PLOT_ENVS := $(subst $(comma),$(space),$(ENV))

define sweep
	@mkdir -p logs
	@echo "  $(1) on $(ENV): $(words $(SEEDS)) seeds, $(JOBS) at a time -> logs/$(1)-$(ENV)-seed*.log"
	@echo "$(SEEDS)" | tr ' ' '\n' | xargs -P $(JOBS) -I{} sh -c \
	  'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(TRAIN) --multirun \
	     hydra.sweep.dir=$(SWEEPS)/$(1)/$(ENV)/seed$$1 \
	     env=$(ENV) total_timesteps=$(TIMESTEPS) seed=$$1 $(PIN) $(2) $(EXTRA) \
	     > logs/$(1)-$(ENV)-seed$$1.log 2>&1' sh {}
	@echo "  done -> $(SWEEPS)/$(1)/$(ENV)/"
endef

# --- Help ---------------------------------------------------------------------
.DEFAULT_GOAL := help

help:                        ## list the targets worth calling
	@grep -hE '^[a-z][a-z0-9-]*:.*##' $(MAKEFILE_LIST) \
	  | awk -F':.*##' '{printf "  %-18s %s\n", $$1, $$2}'

# --- Setup -------------------------------------------------------------------

install:                     ## install the package, dev+hpo extras and git hooks
	uv pip install -e ".[dev,hpo]"
	pre-commit install

# --- Code quality ------------------------------------------------------------

format:                      ## auto-fix formatting and lint
	ruff format src tests scripts
	ruff check --fix src tests scripts

check:                       ## verify formatting and lint without changing files
	ruff format --check src tests scripts
	ruff check src tests scripts

pre-commit:
	pre-commit run --all-files

test:                        ## unit tests
	pytest tests/ -v

sanity:                      ## one real training run on the trivial env
	$(TRAIN) env=empty_5x5 bonus=none seed=0 total_timesteps=50_000

# --- Prerequisites for any LLM condition -------------------------------------
# Run these IN ORDER before the first llm/rnd_llm sweep on a new environment.

# Is env cacheable at all? `plan_classes` must stay bounded and `coverage` near
# 1.0, or the cache never closes and most episodes train with no bonus (KeyCorridor:
# 5,260 classes, 13% coverage -- unusable). Costs no API budget; run it first.
AUDIT_ENV ?= MiniGrid-MultiRoom-N4-S5-v1
audit:                       ## is AUDIT_ENV cacheable at all? (new environments only)
	$(PYTHON) -c "from $(PKG).bonus.llm import audit_env; print(audit_env('$(AUDIT_ENV)'))"


# Gets one plan per plan class via the LLM and appends it to llm.cache_path.
warm-cache:                  ## buy plans for a NEW env; the shipped cache already covers the study
	$(TRAIN) --multirun env=$(ENV) bonus=rnd_llm seed=0 \
	  total_timesteps=20_000 $(PIN) llm.cache_only=false


# --- Conditions ---------------------------------------------------------------
run-ppo:                     ## no bonus
	$(call sweep,ppo,bonus=none)

run-rnd:                     ## BASELINE
	$(call sweep,rnd,bonus=rnd)

run-llm:                     ## does the LLM add anything beyond novelty?
	$(call sweep,llm,bonus=llm)

run-rnd-llm:                 ## constant beta_llm
	$(call sweep,rnd_llm,bonus=rnd_llm)

# Warm start: beta_llm decays to 0 over the first anneal_llm_frac of the run.
run-warmstart:               ## beta_llm annealed to 0 over the first fraction of the run
	$(call sweep,warmstart,bonus=rnd_llm_warmstart)


# --- Controls ------------------------------------------------------------------
# Same subgoals and same total bonus, order permuted: separates whether the LLM picked
# the right targets from whether it sequenced them usefully.
run-shuffled:                ## CONTROL: plan order permuted
	$(call sweep,rnd_llm_shuffled,bonus=rnd_llm llm.shuffle_plan=true)

# Add-on, not in $(CONDITIONS) -- see HANDOFF.md. Run it only IF PPO+RND+LLM plateaus
# below PPO+RND, to separate the two explanations: shaping shrinking the normalized
# advantage at a fixed ent_coef, or the shaped MDP having a different optimum. Lowering
# ent_coef undoes only the first. The value assumes beta_llm=0.5; rescale if it moves.
run-entcoef:                 ## add-on, not in the study: scale-matched ent_coef
	$(call sweep,rnd_llm_ent003,bonus=rnd_llm ent_coef=0.003)

# LockedRoom: the scoped negative, at 6.7x the MultiRoom budget so a null bounds the
# method rather than merely under-running it. Returns may well stay 0 throughout -- then
# the reportable signal is subgoals_completed and policy entropy, not return.
LOCKED_SEEDS ?= 0 21 42 63 84

run-lockedroom:              ## scoped negative: LockedRoom, 5M steps, 3 conditions
	$(MAKE) run-rnd run-rnd-llm run-warmstart ENV=lockedroom \
	  TIMESTEPS=5_000_000 SEEDS="$(LOCKED_SEEDS)"

# --- Sensitivity ---------------------------------------------------------------
# Each condition's own coefficient, swept over a log grid and reported as a curve.
BETA_LLM    ?= 0.125 0.25 0.5 0.75 1.0
BETA_RND    ?= 0.0001 0.0003 0.001 0.003 0.01
# Selection seeds, disjoint from $(SEEDS) so a coefficient is never chosen on the seeds
# it is later reported on. Bands: 0..189 reported, 5xx selection, 210+ HPO search.
SWEEP_SEEDS ?= 501 502 503 504 505

# Sensitivity output goes to its OWN root so `make plots` doesnt read it.
SENS_SWEEPS ?= multirun_sens

define betasweep
	@mkdir -p logs
	@echo "  $(1): $(words $(2)) values x $(words $(SWEEP_SEEDS)) seeds, $(JOBS) at a time"
	@for b in $(2); do for s in $(SWEEP_SEEDS); do echo "$$b $$s"; done; done | \
	  xargs -P $(JOBS) -n2 sh -c \
	    'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(TRAIN) --multirun \
	       hydra.sweep.dir=$(SENS_SWEEPS)/$(1)/$(ENV)/b$$1-seed$$2 \
	       env=$(ENV) total_timesteps=$(TIMESTEPS) seed=$$2 $(3)=$$1 $(PIN) $(4) \
	       > logs/$(1)-$(ENV)-b$$1-seed$$2.log 2>&1' sh
	@echo "  done -> $(SENS_SWEEPS)/$(1)/$(ENV)/"
endef

sweep-beta-llm:              ## sensitivity of beta_llm
	$(call betasweep,sens_beta_llm,$(BETA_LLM),llm.beta,bonus=rnd_llm)

sweep-beta-rnd:              ## sensitivity of beta_rnd
	$(call betasweep,sens_beta_rnd,$(BETA_RND),rnd.beta,bonus=rnd)


# --- Analysis -----------------------------------------------------------------

# Reads every sweep under $(SWEEPS)/ and plots the graphs.
plots:                       ## rliable figures for ENV, from the sweep dir
	$(PYTHON) -m $(PKG).analysis.plots --run-dir $(SWEEPS) \
	  --env $(PLOT_ENVS) --reps 10_000

tensorboard:                 ## browse the runs
	tensorboard --logdir $(SWEEPS)

# --- Full study ---------------------------------------------------------------
# Phase 1 picks the coefficients; set them in configs/config.yaml, then run phase 2,
# which produces the reported numbers.
SENSITIVITY := sweep-beta-llm sweep-beta-rnd
CONDITIONS  := run-ppo run-rnd run-llm run-rnd-llm run-warmstart run-shuffled

sensitivity: $(SENSITIVITY)  ## phase 1: beta sweeps on the selection seeds
conditions: $(CONDITIONS)    ## phase 2: every condition + both controls, at the chosen betas

# Re-runs a study whose betas are already committed, so the two phases can go back to
# back. LockedRoom is plotted on its own -- its return ceiling differs, so pooling it
# with MultiRoom would be meaningless.
reproduce: sensitivity conditions run-lockedroom  ## all three phases, then figures (~8h)
	$(MAKE) plots
	$(MAKE) plots ENV=lockedroom

# --- Cleanup -------------------------------------------------------------------
clean:                       ## drop __pycache__, .pytest_cache and .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +


# --- HPO (optional; not part of the study) --------------------------------------
# One DEHB search per condition, sequential; each already spends $(JOBS) workers.
#   make hpo HPO_CONDS=rnd_llm      a single search
#   make hpo HPO_CONDS="ppo rnd llm rnd_llm rnd_llm_warmstart"    everything
HPO_CONDS ?= rnd rnd_llm_warmstart rnd_llm

hpo:                         ## run DEHB searches (see HPO_CONDS)
	@mkdir -p logs
	@echo "commit $$(git rev-parse --short HEAD 2>/dev/null), \
	  $$(git status --porcelain | wc -l | tr -d ' ') files dirty"
	@for c in $(HPO_CONDS); do \
	  echo "=== $$c ==="; \
	  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(TRAIN) --multirun --config-name hpo/$$c \
	    > logs/hpo-$$c.log 2>&1; \
	done

hpo-incumbents:              ## print the incumbent of each finished search
	@for c in $(HPO_CONDS); do \
	  echo "--- $$c ---"; \
	  if [ ! -f logs/hpo-$$c.log ]; then echo "  (not started)"; else \
	    out=$$(grep -A6 "incumbent configuration" logs/hpo-$$c.log | tail -7); \
	    if [ -n "$$out" ]; then echo "$$out"; else echo "  (no incumbent yet)"; fi; \
	  fi; \
	done
