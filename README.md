# TEMPPO: TEmplate-Matched Proposals for PPO
**TEMPPO** (**TE**mplate-**M**atched **P**roposals for P**PO**) tries to accelerate PPO in sparse-reward MiniGrid environments by having an LLM fill a fixed subgoal template from the environment map, then rewarding the agent for completing those proposed subgoals during training.

PPO adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl)'s `ppo_atari.py` to
sparse-reward [MiniGrid](https://github.com/Farama-Foundation/Minigrid), then extended
with two exploration bonuses:

* **RND** — a task-agnostic novelty bonus that drives discovery, from the prediction
  error of a trained network against a fixed random target.
* **LLM subgoals** — an LLM proposes subgoals for a layout; the agent is rewarded for
  completing them.

Both enter through the reward, so two coefficients define every condition:

```
r = r_env + beta_rnd * r_rnd + beta_llm * r_llm
```

The question is whether LLM subgoals add exploration signal **beyond** what
task-agnostic novelty already provides.

References: PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)),
RND ([Burda et al., 2018](https://arxiv.org/abs/1810.12894)),
CleanRL ([Huang et al., 2022](https://jmlr.org/papers/v23/21-1342.html)).

## Installation
1. Clone this repository:
    * ``git clone https://github.com/Ole-rgb/TEMPPO.git``
2. Install the [uv](https://docs.astral.sh/uv/) package manager:
    * ``pip install uv``
3. Create a virtual environment:
    * ``uv venv --python 3.11``
4. Activate it:
    * ``source .venv/bin/activate``
5. Install the project (editable install + dev tools + HPO + pre-commit hooks):
    * ``make install``

## Dry run (check everything works)
Before starting anything long, train on the trivial env. This exercises the whole
pipeline — training, evaluation, and every output artifact — in about 15 seconds:
```
python -m rl_final.ppo env=empty_5x5 total_timesteps=20000
```
Expect `final eval_return` close to **0.955**, which is optimal for `Empty-5x5`
(`1 - 0.9 * 5/100`, i.e. the 5-step path). A value near `0.0` means something is
broken. `make sanity` runs the same thing at 50k steps.

> Always launch through `rl_final.ppo`, never `rl_final.ppo.ppo_minigrid`.
> Running the module directly executes it as `__main__`, and cloudpickle then
> serializes the task function *by value*, so every parallel launcher (and every
> HPO sweep) dies on `cannot pickle 'CudnnModule'`. Single runs work either way,
> which is what makes it easy to miss.

A single run lands in `outputs/<date>/<time>/`. Sweeps go to `multirun/` instead --
the `make run-*` targets pin `multirun/<condition>/<env>/seed<N>/`:
```
<run_dir>/
├── .hydra/config.yaml          fully resolved config for this run
├── ppo_minigrid.log
├── eval_rewards.csv            one row per eval; see the header for columns
├── agent.pt                    final weights
└── events.out.tfevents.*       TensorBoard scalars
```
Inspect the curves with `tensorboard --logdir outputs` (or `make tensorboard`, which
covers both sweep roots). If the CLI dies on `No module named 'pkg_resources'`, your
setuptools is >= 81; `uv pip install "setuptools<81"` restores it.

Of these, only `eval_rewards.csv` and `.hydra/config.yaml` are committed -- together
~1.5 MB for the whole study, which is what lets `make plots` run from a fresh clone.
Weights, TensorBoard events and videos are gitignored (~600 MB), so archive `multirun/`
yourself if you want the scalars the CSV does not carry.

## Recording videos
Add `capture_video=true` to any run:
```
python -m rl_final.ppo env=doorkey_8x8 capture_video=true
```
You get one mp4 per evaluation, each in a step-named folder so successive evals
never overwrite each other:
```
<run_dir>/videos/step_00005120/rl-video-episode-0.mp4
                 step_00010240/rl-video-episode-0.mp4
                 ...
```
Watching them in order shows how the policy changes over training. Only episode 0
of each eval is recorded, and eval episodes use fixed seeds, so every video shows
the *same* layout — differences between them are the policy, not luck. Use
`eval_interval` to control how often a video is produced.

> Keep `capture_video=false` for the real sweeps. Rendering a 640-step DoorKey
> episode to mp4 on every evaluation is not free.

## Verify the install
Cheapest first. Run them after any environment change:
```
make test            # 116 unit tests, ~8 s
make sanity          # one real training run on the trivial env
pytest -m slow       # 10 end-to-end tests that actually train, ~2 min
```
The slow ones are excluded from `make test` by `addopts` in `pyproject.toml`; run them
before a sweep, since they are what pins the eval-CSV schema and reproducibility.

## Reproduce the study

Everything below runs from a fresh clone with **no API key and no LLM spend** — the plan
cache ships filled for all three environments used here. Run the steps in order; each is
one command.

### Step 1 — coefficient sweeps (75 runs, ~3.2 h)

```
make sensitivity
make plots-sensitivity
```

Three sweeps on the selection seeds into `multirun_sens/`: `beta_rnd` on PPO+RND and
`beta_llm` on PPO+LLM choose the coefficients; a `(beta_rnd, beta_llm)` grid checks that
transplanting them was safe. Read the figures, then set the chosen values in
`configs/config.yaml`:

```yaml
rnd:
  beta: 0.001
llm:
  beta: 0.25
```

Those are the values this study selected. If you are reproducing rather than re-deriving,
they are already in the config and you can go straight to step 2.

### Step 2 — conditions on MultiRoom (60 runs, ~2.6 h)

```
make conditions
```

PPO, +RND, +LLM, +RND+LLM, warm start, and the shuffled-plan control, at 750k steps on
seeds 0…189.

### Step 3 — conditions on DoorKey (60 runs, ~3.3 h)

```
make run-doorkey
```

The same six conditions, same budget, same seeds, same coefficients — only the task
changes. The coefficients are transplanted unchanged from MultiRoom, so neither method
gets per-task tuning.

### Step 4 — LockedRoom, the scoped negative (15 runs, ~4.3 h)

```
make run-lockedroom
```

PPO+RND, PPO+RND+LLM and the warm start at 5M steps on 5 seeds. Nothing moves before ~4M,
which is why the budget is 6.7x the MultiRoom one.

### Step 5 — every figure

```
make plots-sensitivity
make plots
make plots ENV=doorkey_8x8
make plots ENV=lockedroom

# probability-of-improvement panels against a non-default baseline
make plots BASELINE="PPO+RND"                    # does the LLM add anything beyond novelty?
make plots BASELINE="PPO+LLM"                    # does anything improve on plain subgoals?
make plots ENV=doorkey_8x8 BASELINE="PPO+LLM"    # the same, on the second environment

# diagnostics: why a result happened, not what happened
make plots ENV=lockedroom COLUMN=entropy         # the policy collapse behind the LockedRoom null
```

Figures land in `figures/<sweep>/<env>/`, one directory per (sweep, environment) pair, so
sweeps never collide. Re-running the same sweep does rewrite its own files.

`THRESHOLD` is the return counted as "solved" in the performance profile, and scales with
what each env can reach: 0.70 MultiRoom, 0.80 DoorKey, 0.5 LockedRoom (nothing there gets
near its ceiling). The Makefile picks it per env -- override only to test sensitivity.

`REPS` (default 10,000) is the bootstrap resample count. It does not narrow the intervals --
seed count sets their width -- so raising it only stabilises the printed endpoints, at about
50 min per curve figure at 50k. Use `REPS=2000` while iterating.

`BASELINE` is the reference each `P(X > Y)` panel is drawn against. It defaults to plain PPO
where that arm exists. LockedRoom has no PPO arm, so the Makefile names one per env there -- otherwise the fallback would pick alphabetically rather than the arm that actually works.
`plots-sensitivity` passes its own for the same reason.

A non-default baseline is written as `probability_of_improvement_<baseline>.png` beside the default panel.

The return says what happened; the other columns of `eval_rewards.csv` say why. `entropy`,
`subgoals_completed` and `episodic_length` get the same IQM-with-bootstrap-CI treatment:

```
make plots ENV=lockedroom COLUMN=subgoals_completed
make plots ENV=doorkey_8x8 COLUMN=episodic_length
```

Never pool environments in one figure: return ceilings differ (MultiRoom ~0.78, DoorKey
~0.97), so a pooled aggregate tracks the easier task and hides that a bonus can help on one
and hurt on another.

### Or all of it at once

```
make reproduce          # steps 1-5 back to back (~13 h at JOBS=6)
```

This assumes the coefficients are already committed in `configs/config.yaml`, which they
are. Doing the study for the *first* time means stopping after step 1 to read the curves —
otherwise the coefficients are fixed before the sweep meant to inform them.

## The plan cache

**For the three environments in this study you do not need to do anything.**
`scripts/subgoal_cache_claude-opus-4-8.jsonl` is committed and ships filled — 11 plan
classes covering MultiRoom-N4-S5 (7), LockedRoom (3) and DoorKey-8x8 (1) — so every step
above runs from a fresh clone with no API key and no spend. Skip to the next section.

The rest of this applies only if you add **a new environment**. `make warm-cache` buys
one plan per plan class. Run it single-process, never in parallel, or concurrent jobs
each query the LLM for the same uncached class and train against different answers.

Every sweep pins `llm.cache_only=true` (see `PIN` in the Makefile) so a run can never
buy a plan mid-flight: six parallel jobs missing the same class would each get a different
answer and append to the same file.

That makes verification necessary, because the failure is silent. `cache_only=true` stops
API calls but not a zero-bonus run: on a cache miss `SubgoalTracker._replan` binds an empty plan and
training continues. The LLM arm then trains as plain PPO+RND while every other log line
looks healthy. After warming a new env, read the scalars back:
```
python - <<'EOF'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
f = sorted(glob.glob('multirun/**/events.out.tfevents.*', recursive=True))[-1]
ea = EventAccumulator(f); ea.Reload()
for t in ['llm/plan_cache_misses', 'llm/plan_classes_cached', 'llm/subgoals_completed']:
    if t in ea.Tags()['scalars']:
        print(f'{t:28s} {ea.Scalars(t)[-1].value}')
EOF
```
`plan_cache_misses` must be **0** and `subgoals_completed` must be **> 0**; if not, warm
the cache again before launching.

## Hyperparameter optimization
One DEHB search per condition, defined in `configs/hpo/`. Each condition gets its own
search space under the same `n_trials`, seeds and objective, and incumbents are compared
afterwards. Every hyperparameter is therefore either taken unchanged from a published
reference implementation and held identical across conditions, or searched by DEHB.

The PPO defaults in `configs/config.yaml` are CleanRL's `ppo.py` values and are not
MiniGrid-tuned; no published PPO configuration solves MultiRoom-N4-S5 or LockedRoom.

| condition | config | search space |
|---|---|---|
| PPO | `hpo/ppo` | `learning_rate`, `ent_coef` |
| PPO+RND | `hpo/rnd` | `+ rnd.beta` |
| PPO+LLM | `hpo/llm` | `+ llm.beta` |
| PPO+RND+LLM | `hpo/rnd_llm` | `+ rnd.beta`, `llm.beta` |
| PPO+RND+LLM warm | `hpo/rnd_llm_warmstart` | `+ rnd.beta`, `llm.beta` |

Spaces differ in size (2D–4D) at equal `n_trials`, so the baselines get more coverage
per dimension than the proposed method.

Seeds are split three ways: `0…189` for reported runs, `501…505` for coefficient
selection, `210…294` for the HPO search. They stay disjoint so nothing is selected on
the seeds it is later reported on. Search output is not a reportable number — re-run
each incumbent as a normal sweep and take the figures from that.

### Running
```
make hpo                                  # the three headline conditions  (~9h)
make hpo HPO_CONDS=rnd_llm                # a single search               (~3h)
make hpo HPO_CONDS="ppo rnd llm rnd_llm rnd_llm_warmstart"   # everything (~16h)
make hpo-incumbents                       # incumbent of each finished search
```
Searches run one at a time — each already spends `JOBS=6` workers, so two at once only
oversubscribe the machine.

Anything multi-hour (`make hpo`, `make reproduce`) should survive an idle machine:
```
mkdir -p logs   # gitignored, so a fresh clone has none and the redirect would fail
caffeinate -is nohup make reproduce > logs/study.log 2>&1 &
```
Keep it plugged in and **the lid open** — `caffeinate` does not override clamshell sleep.

Shared settings live in `configs/hpo/_base.yaml`: equal `n_trials`, equal seeds, equal
objective and equal fidelities for every condition, in one place so they cannot drift.
A condition file adds only its bonus group and its own coefficient(s).
