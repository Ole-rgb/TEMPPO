"""
Hydra run dirs -> Run objects.

    <root>/<...>/.hydra/config.yaml     the run's fully-specified config
    <root>/<...>/eval_rewards.csv       written straight into the job dir

`root` is one output dir -- one `--multirun` sweep or one run. A sweep emits each
(env, condition, seed).
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rl_final.bonus.llm import resolve_anneal_steps

BONUS_NAMES = {
    "none": "PPO",
    "rnd": "PPO+RND",
    "llm": "PPO+LLM",
    "rnd_llm": "PPO+RND+LLM",
    "rnd_llm_warmstart": "PPO+RND+LLM (warm)",
}
# KEEP IN SYNC WITH configs/config.yaml -- test_run_loader asserts they match.
TUNABLE_DEFAULTS = {
    "ent_coef": 0.01,
    "learning_rate": 2.5e-4,
    "vf_coef": 0.5,
    "clip_coef": 0.2,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "update_epochs": 4,
    "num_minibatches": 4,
}


@dataclass(frozen=True)
class Run:
    """One seed of one condition."""

    run_dir: Path
    env: str
    condition: str
    seed: int
    total_timesteps: int
    steps: np.ndarray
    returns: np.ndarray


def condition_label(cfg) -> str:
    """Human-readable condition name, including any coefficient that is active."""
    base = BONUS_NAMES.get(cfg.bonus.name, cfg.bonus.name)
    beta_rnd = float(cfg.bonus.beta_rnd)
    beta_llm = float(cfg.bonus.beta_llm)
    # Resolved, not raw: a run configured with anneal_llm_frac must land on the same
    # label as the equivalent anneal_llm_steps run, or the two forms split one
    # condition in two.
    anneal = resolve_anneal_steps(cfg)
    skip_goal = bool((cfg.get("llm") or {}).get("skip_go_to_goal", False))
    shuffled = bool((cfg.get("llm") or {}).get("shuffle_plan", False))

    parts = []
    if beta_rnd:
        parts.append(f"β_rnd={beta_rnd:g}")
    if beta_llm:
        decay = f"→0@{anneal / 1000:g}k" if anneal > 0 else ""
        parts.append(f"β_llm={beta_llm:g}{decay}")
        if skip_goal:
            parts.append("no-goal")
        if shuffled:
            parts.append("shuffled")
        # Redundant with beta_llm -- only their product reaches the reward -- so it
        # is pinned at 1.0 and beta_llm is tuned alone. Tagged if that ever slips.
        bonus_scale = float((cfg.get("llm") or {}).get("subgoal_bonus", 1.0))
        if bonus_scale != 1.0:
            parts.append(f"sg={bonus_scale:g}")

    # Tuned PPO hyperparameters, appended only when they leave their default.
    for key, default in TUNABLE_DEFAULTS.items():
        value = cfg.get(key, default)
        if value is not None and float(value) != float(default):
            parts.append(f"{key}={float(value):g}")

    if not bool(cfg.get("anneal_lr", True)):
        parts.append("const-lr")
    horizon = cfg.get("schedule_timesteps", None)
    if horizon and int(horizon) != int(cfg.get("total_timesteps", horizon)):
        parts.append(f"lr-horizon={int(horizon) / 1000:g}k")

    return f"{base} ({', '.join(parts)})" if parts else base


def load_eval_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read eval_rewards.csv -> (steps, returns)."""
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    if raw.size == 0:
        raise ValueError(f"no eval rows in {path}")
    raw = np.atleast_2d(raw)
    return raw[:, 0].astype(np.int64), raw[:, 1]


def eval_csv_path(run_dir: Path) -> Path | None:
    """The CSV belonging to one Hydra job, or None if it never wrote one."""
    flat = run_dir / "eval_rewards.csv"
    return flat if flat.exists() else None


def discover_runs(roots: list[Path]) -> list[Run]:
    """Find every eval_rewards.csv under `roots` and tag it with its config."""
    runs: list[Run] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for cfg_path in sorted(root.rglob(".hydra/config.yaml")):
            run_dir = cfg_path.parent.parent
            try:
                cfg = OmegaConf.load(cfg_path)
                OmegaConf.resolve(cfg)
            except Exception as exc:
                print(f"  ! skipping {run_dir}: unreadable config ({exc})")
                continue

            if csv_path := eval_csv_path(run_dir):
                try:
                    steps, returns = load_eval_csv(csv_path)
                except ValueError as exc:  # died before its first eval
                    print(f"  ! skipping {csv_path.parent}: {exc}")
                    continue
                runs.append(
                    Run(
                        run_dir=run_dir,
                        env=cfg.env.name,
                        condition=condition_label(cfg),
                        seed=int(cfg.seed),
                        total_timesteps=int(cfg.total_timesteps),
                        steps=steps,
                        returns=returns,
                    )
                )
    return runs


def check_unique_seeds(runs: list[Run]) -> None:
    """Assert one run per (env, condition, seed); raise naming both offenders.

    Hydra cannot collide inside one sweep, so this fires when the roots overlap --
    either a parent of several sweeps, or two sweeps that repeat a condition.
    """
    seen: dict[tuple[str, str, int], Path] = {}
    for run in runs:
        key = (run.env, run.condition, run.seed)
        if key in seen:
            raise ValueError(
                f"two runs share (env={run.env}, condition={run.condition}, "
                f"seed={run.seed}):\n    {seen[key]}\n    {run.run_dir}\n"
                f"pass sweep dirs that do not overlap, not a parent of several"
            )
        seen[key] = run.run_dir


def load(roots: Path | str | Iterable[Path | str], env=None) -> list[Run]:
    """Load one or more sweeps: discover, check for seed collisions, filter, sort.

    Each root is a Hydra output dir -- `multirun/<date>/<time>/` for a sweep or
    `outputs/<date>/<time>/` for a single run. Passing several merges them, which is
    how you compare conditions that were swept separately (a PPO baseline from one
    day against PPO+RND from another) without copying directories around. They must
    not overlap; `check_unique_seeds` enforces that.
    """
    if isinstance(roots, str | Path):
        roots = [roots]
    runs = discover_runs([Path(r) for r in roots])
    check_unique_seeds(runs)
    if env is not None:
        wanted = {env} if isinstance(env, str) else set(env)
        runs = [r for r in runs if r.env in wanted]
    return sorted(runs, key=lambda r: (r.env, r.condition, r.seed))
