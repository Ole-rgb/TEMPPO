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
task-agnostic novelty already provides:

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
5. Install the project (editable install + dev tools + pre-commit hooks):
    * ``make install``

## Dry run (check everything works)
Before starting anything long, train on the trivial env. This exercises the whole
pipeline — training, evaluation, and every output artifact — in about 15 seconds:
```
python -m rl_final.ppo.ppo_minigrid env=empty_5x5 total_timesteps=20000
```
Expect `final eval_return` close to **0.955**, which is optimal for `Empty-5x5`
(`1 - 0.9 * 5/100`, i.e. the 5-step path). A value near `0.0` means something is
broken. For the full three-seed check (`0, 21, 42`), run `make sanity`.

Results land in `outputs/<date>/<time>/`, or `outputs/<sweep>/<job>/` under `--multirun`:
```
<run_dir>/
├── .hydra/config.yaml          fully resolved config for this run
├── ppo_minigrid.log
└── seed_0/
    ├── eval_rewards.csv        columns: eval_steps, eval_rewards
    ├── agent.pt                final weights
    └── events.out.tfevents.*   TensorBoard scalars
```
Inspect the curves with `tensorboard --logdir outputs`.

## Recording videos
Add `capture_video=true` to any run:
```
python -m rl_final.ppo.ppo_minigrid env=doorkey_8x8 capture_video=true
```
You get one mp4 per evaluation, each in a step-named folder so successive evals
never overwrite each other:
```
<run_dir>/seed_0/videos/step_00005120/rl-video-episode-0.mp4
                        step_00010240/rl-video-episode-0.mp4
                        ...
```
Watching them in order shows how the policy changes over training. Only episode 0
of each eval is recorded, and eval episodes use fixed seeds, so every video shows
the *same* layout — differences between them are the policy, not luck. Use
`eval_interval` to control how often a video is produced.

> Keep `capture_video=false` for the real sweeps. Rendering a 640-step DoorKey
> episode to mp4 on every evaluation is not free.

## Reproduce the study
Run all experiments and regenerate the figures with one command:
```
make reproduce
```
