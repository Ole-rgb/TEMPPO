from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from rl_final.analysis.run_loader import (
    TUNABLE_DEFAULTS,
    condition_label,
    discover_runs,
    load,
    load_eval_csv,
)

# Mirrors the real .hydra/config.yaml: interpolations are dumped UNRESOLVED.
CONFIG_TEMPLATE = """
env:
  name: {env}
bonus:
  name: {bonus}
  beta_rnd: {beta_rnd}
  beta_llm: {beta_llm}
seed: {seed}
total_timesteps: {total_timesteps}
rnd:
  beta: 0.1
llm:
  beta: {llm_beta}
"""

BONUS_BETAS = {
    "none": ("0.0", "0.0"),
    "rnd": ("${rnd.beta}", "0.0"),
    "llm": ("0.0", "${llm.beta}"),
    "rnd_llm": ("${rnd.beta}", "${llm.beta}"),
}


def write_run(
    root,
    name,
    rows,
    env="doorkey_8x8",
    bonus="none",
    seed=0,
    llm_beta=0.5,
):
    """Build one Hydra run dir: .hydra/config.yaml + eval_rewards.csv."""
    run_dir = root / name
    (run_dir / ".hydra").mkdir(parents=True)
    beta_rnd, beta_llm = BONUS_BETAS[bonus]
    (run_dir / ".hydra" / "config.yaml").write_text(
        CONFIG_TEMPLATE.format(
            env=env,
            bonus=bonus,
            beta_rnd=beta_rnd,
            beta_llm=beta_llm,
            seed=seed,
            total_timesteps=rows[-1][0] if rows else 0,
            llm_beta=llm_beta,
        )
    )
    csv_path = run_dir / "eval_rewards.csv"
    csv_path.write_text("eval_steps,eval_rewards\n" + "".join(f"{s},{r}\n" for s, r in rows))
    return run_dir


def cfg_for(bonus, llm_beta=0.5):
    cfg = OmegaConf.create(
        CONFIG_TEMPLATE.format(
            env="doorkey_8x8",
            bonus=bonus,
            beta_rnd=BONUS_BETAS[bonus][0],
            beta_llm=BONUS_BETAS[bonus][1],
            seed=0,
            total_timesteps=1000,
            llm_beta=llm_beta,
        )
    )
    OmegaConf.resolve(cfg)
    return cfg


# --- load_eval_csv ----------------------------------------------------------
#
# The reader takes a well-formed CSV at face value: rows land as written, in the
# order the training loop appended them. The only rejected file is one with no
# eval rows at all.


def write_csv(tmp_path, rows):
    """Write a bare eval_rewards.csv (no surrounding run dir) and return its path."""
    path = tmp_path / "eval_rewards.csv"
    path.write_text("eval_steps,eval_rewards\n" + "".join(f"{s},{r}\n" for s, r in rows))
    return path


def test_rows_are_read_in_file_order(tmp_path):
    steps, returns = load_eval_csv(write_csv(tmp_path, [(100, 0.1), (200, 0.2), (300, 0.3)]))
    np.testing.assert_array_equal(steps, [100, 200, 300])
    np.testing.assert_allclose(returns, [0.1, 0.2, 0.3])


def test_header_only_file_raises(tmp_path):
    """A run that never evaluated must not slip through as an empty curve."""
    with pytest.raises(ValueError):
        load_eval_csv(write_csv(tmp_path, []))


def test_single_row_file_is_not_squeezed_to_scalar(tmp_path):
    steps, returns = load_eval_csv(write_csv(tmp_path, [(512, 0.25)]))
    np.testing.assert_array_equal(steps, [512])
    np.testing.assert_allclose(returns, [0.25])


# --- condition_label: interpolations must be resolved -----------------------


def test_sweep_points_get_distinct_labels():
    """The headline sweep varies llm.beta while bonus.name stays 'rnd_llm'.

    If the label ignored the betas (or read the raw '${llm.beta}' string), every
    point of the sweep would collapse onto a single curve.
    """
    labels = {condition_label(cfg_for("rnd_llm", llm_beta=b)) for b in (0.1, 0.25, 0.5, 1.0)}
    assert len(labels) == 4


def test_label_reflects_active_coefficients_only():
    assert condition_label(cfg_for("none")) == "PPO"
    assert condition_label(cfg_for("rnd")) == "PPO+RND (β_rnd=0.1)"
    assert condition_label(cfg_for("llm", 0.5)) == "PPO+LLM (β_llm=0.5)"
    assert condition_label(cfg_for("rnd_llm", 0.25)) == "PPO+RND+LLM (β_rnd=0.1, β_llm=0.25)"


def test_four_conditions_are_all_distinguishable():
    labels = {condition_label(cfg_for(b)) for b in ("none", "rnd", "llm", "rnd_llm")}
    assert len(labels) == 4


# --- check_unique_seeds: a repeated seed is an error, not something to fix ---


def test_repeated_seed_raises_and_names_both_runs(tmp_path):
    """Two dirs at the same (env, condition, seed) would count as two seeds."""
    write_run(tmp_path, "old", [(100, 0.1)], seed=0)
    write_run(tmp_path, "new", [(100, 0.9)], seed=0)

    with pytest.raises(ValueError, match="two runs share") as exc:
        load(tmp_path)
    assert "old" in str(exc.value) and "new" in str(exc.value)


def test_distinct_seeds_are_all_kept(tmp_path):
    for seed in (0, 21, 42):
        write_run(tmp_path, f"s{seed}", [(100, 0.1 * seed)], seed=seed)

    assert [r.seed for r in load(tmp_path)] == [0, 21, 42]


def test_same_seed_different_conditions_is_not_a_collision(tmp_path):
    write_run(tmp_path, "a", [(100, 0.1)], bonus="none", seed=0)
    write_run(tmp_path, "b", [(100, 0.2)], bonus="rnd", seed=0)

    assert len(load(tmp_path)) == 2


def test_same_seed_different_envs_is_not_a_collision(tmp_path):
    write_run(tmp_path, "a", [(100, 0.1)], env="empty_5x5", seed=0)
    write_run(tmp_path, "b", [(100, 0.2)], env="doorkey_8x8", seed=0)

    assert len(load(tmp_path)) == 2


def test_load_filters_by_env(tmp_path):
    """A bare string must not be iterated character-wise -- set("empty_5x5")
    is a set of letters and would match nothing."""
    for env in ("empty_5x5", "doorkey_5x5", "doorkey_8x8"):
        write_run(tmp_path, env, [(100, 0.1)], env=env, seed=0)

    assert [r.env for r in load(tmp_path)] == ["doorkey_5x5", "doorkey_8x8", "empty_5x5"]
    assert [r.env for r in load(tmp_path, env="empty_5x5")] == ["empty_5x5"]
    assert [r.env for r in load(tmp_path, env=["empty_5x5", "doorkey_8x8"])] == [
        "doorkey_8x8",
        "empty_5x5",
    ]


# --- discovery --------------------------------------------------------------


def test_discovery_reads_env_condition_and_seed(tmp_path):
    write_run(tmp_path, "r", [(100, 0.5)], env="multiroom_n4s5", bonus="rnd_llm", seed=42)

    (run,) = discover_runs([tmp_path])
    assert run.env == "multiroom_n4s5"
    assert run.seed == 42
    assert "PPO+RND+LLM" in run.condition


def test_discovery_finds_nested_multirun_job_dirs(tmp_path):
    """--multirun writes multirun/<date>/<time>/<job>/, one config per job."""
    nested = tmp_path / "2026-08-07" / "12-00-00"
    for job, seed in enumerate([0, 21, 42]):
        write_run(nested, str(job), [(100, 0.5)], seed=seed)

    assert len(discover_runs([tmp_path])) == 3


def test_run_without_eval_rows_is_skipped(tmp_path):
    write_run(tmp_path, "crashed", [], seed=0)
    write_run(tmp_path, "ok", [(100, 0.5)], seed=21)

    assert len(discover_runs([tmp_path])) == 1


def test_missing_root_is_not_an_error(tmp_path):
    assert discover_runs([tmp_path / "does_not_exist"]) == []


# --- output layout: artifacts sit in the job dir, seed comes from the config ---


def test_csv_is_read_from_the_job_dir_itself(tmp_path):
    run_dir = write_run(tmp_path, "r", [(100, 0.5)], seed=42)

    assert (run_dir / "eval_rewards.csv").exists(), "csv should not be in a subdir"
    assert not list(run_dir.glob("seed_*"))
    (run,) = discover_runs([tmp_path])
    assert run.seed == 42


def test_seed_comes_from_config_not_the_path(tmp_path):
    """The whole reason seed_<N>/ is droppable: .hydra/config.yaml already has it.

    Renaming the job dir to something meaningless must not change the seed.
    """
    write_run(tmp_path, "0", [(100, 0.5)], seed=63)

    (run,) = discover_runs([tmp_path])
    assert run.seed == 63


def test_seed_subdir_csvs_are_ignored(tmp_path):
    """Only the job dir is read: a seed_<N>/ CSV is not a run this layer knows."""
    run_dir = tmp_path / "old"
    (run_dir / ".hydra").mkdir(parents=True)
    (run_dir / ".hydra" / "config.yaml").write_text(
        CONFIG_TEMPLATE.format(
            env="doorkey_8x8",
            bonus="none",
            beta_rnd="0.0",
            beta_llm="0.0",
            seed=21,
            total_timesteps=100,
            llm_beta=0.5,
        )
    )
    (run_dir / "seed_21").mkdir()
    (run_dir / "seed_21" / "eval_rewards.csv").write_text("eval_steps,eval_rewards\n100,0.5\n")

    assert discover_runs([tmp_path]) == []


def test_job_dir_without_a_csv_is_skipped(tmp_path):
    """A run that died before writing any CSV has a config but no eval file."""
    run_dir = tmp_path / "crashed"
    (run_dir / ".hydra").mkdir(parents=True)
    (run_dir / ".hydra" / "config.yaml").write_text(
        CONFIG_TEMPLATE.format(
            env="doorkey_8x8",
            bonus="none",
            beta_rnd="0.0",
            beta_llm="0.0",
            seed=0,
            total_timesteps=100,
            llm_beta=0.5,
        )
    )

    assert discover_runs([tmp_path]) == []


def test_anneal_horizon_is_part_of_the_condition_label():
    """A schedule study varies only the horizon, so the label has to carry it.

    Without this the two horizons share a name and `check_unique_seeds` reads the sweep
    as one condition with every seed run twice.
    """
    cfg = OmegaConf.create(
        {"bonus": {"name": "rnd_llm_warmstart", "beta_rnd": 0.001, "beta_llm": 0.5}}
    )
    labels = set()
    for horizon in (150_000, 300_000):
        cfg.bonus.anneal_llm_steps = horizon
        labels.add(condition_label(cfg))
    assert labels == {
        "PPO+RND+LLM(warm) (β_rnd=0.001, β_llm=0.5→0@150k)",
        "PPO+RND+LLM(warm) (β_rnd=0.001, β_llm=0.5→0@300k)",
    }


def test_runs_recorded_before_warm_start_existed_still_label():
    """The 750k sweeps on disk have no `anneal_llm_steps`; they must keep their names."""
    cfg = OmegaConf.create({"bonus": {"name": "rnd_llm", "beta_rnd": 0.001, "beta_llm": 0.5}})
    assert condition_label(cfg) == "PPO+RND+LLM (β_rnd=0.001, β_llm=0.5)"
    cfg.bonus.anneal_llm_steps = 0
    assert condition_label(cfg) == "PPO+RND+LLM (β_rnd=0.001, β_llm=0.5)"


def test_dropping_the_terminal_subgoal_is_part_of_the_condition_label():
    """Same betas, different plan -- the label has to separate them or they collide."""
    cfg = OmegaConf.create(
        {"bonus": {"name": "rnd_llm", "beta_rnd": 0.001, "beta_llm": 0.5}, "llm": {"beta": 0.5}}
    )
    assert condition_label(cfg) == "PPO+RND+LLM (β_rnd=0.001, β_llm=0.5)"
    cfg.llm.skip_go_to_goal = True
    assert condition_label(cfg) == "PPO+RND+LLM (β_rnd=0.001, β_llm=0.5, no-goal)"


def test_shuffled_plan_is_part_of_the_condition_label():
    """The control differs from rnd_llm only here; without the tag they collide."""
    cfg = OmegaConf.create(
        {"bonus": {"name": "rnd_llm", "beta_rnd": 0.001, "beta_llm": 0.5}, "llm": {"beta": 0.5}}
    )
    assert condition_label(cfg) == "PPO+RND+LLM (β_rnd=0.001, β_llm=0.5)"
    cfg.llm.shuffle_plan = True
    assert condition_label(cfg) == "PPO+RND+LLM (β_rnd=0.001, β_llm=0.5, shuffled)"


def test_skip_go_to_goal_is_ignored_without_an_llm_bonus():
    """No LLM bonus means no plan is bound, so the flag must not split the condition."""
    cfg = OmegaConf.create(
        {
            "bonus": {"name": "rnd", "beta_rnd": 0.001, "beta_llm": 0.0},
            "llm": {"beta": 0.5, "skip_go_to_goal": True},
        }
    )
    assert condition_label(cfg) == "PPO+RND (β_rnd=0.001)"


# --- condition_label must cover every tuned hyperparameter ------------------
# HPO varies these. If one is missing from the label, two incumbents differing
# only in it become the same condition and check_unique_seeds calls them
# duplicate seeds. This bug has already shipped twice.


def test_tunable_defaults_match_the_real_config():
    """TUNABLE_DEFAULTS is a hand-copy of configs/config.yaml; pin them together.

    If they drift, a run at the config default gets tagged as if it were tuned,
    silently splitting one condition into two.
    """
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "config.yaml")
    for key, default in TUNABLE_DEFAULTS.items():
        assert key in cfg, f"{key} is not in config.yaml"
        assert float(cfg[key]) == float(default), (
            f"{key}: config.yaml has {cfg[key]}, TUNABLE_DEFAULTS has {default}"
        )


def test_default_values_do_not_change_existing_labels():
    """Pre-HPO runs must keep the names they already have."""
    assert condition_label(cfg_for("rnd")) == "PPO+RND (β_rnd=0.1)"
    cfg = cfg_for("rnd")
    cfg.ent_coef = 0.01  # the default, explicitly present
    assert condition_label(cfg) == "PPO+RND (β_rnd=0.1)"


def test_ent_coef_separates_otherwise_identical_conditions():
    """The exact collision the ent_coef control would have hit."""
    a, b = cfg_for("rnd_llm", 0.5), cfg_for("rnd_llm", 0.5)
    a.ent_coef, b.ent_coef = 0.01, 0.003
    assert condition_label(a) != condition_label(b)
    assert "ent_coef=0.003" in condition_label(b)


def test_every_tunable_separates_conditions():
    """No tuned hyperparameter may be invisible to the label."""
    for key, default in TUNABLE_DEFAULTS.items():
        base, tuned = cfg_for("rnd_llm", 0.5), cfg_for("rnd_llm", 0.5)
        tuned[key] = float(default) * 2 + 1  # guaranteed different
        assert condition_label(base) != condition_label(tuned), f"{key} is invisible"


def test_subgoal_bonus_is_tagged_when_unpinned():
    """beta_llm and subgoal_bonus are redundant; only their product matters."""
    cfg = cfg_for("rnd_llm", 0.5)
    cfg.llm.subgoal_bonus = 2.0
    assert "sg=2" in condition_label(cfg)


def test_constant_lr_is_a_distinct_condition():
    """anneal_lr changes what the run optimizes without changing any coefficient."""
    a, b = cfg_for("rnd_llm", 0.5), cfg_for("rnd_llm", 0.5)
    a.anneal_lr, b.anneal_lr = True, False
    assert condition_label(a) != condition_label(b)
    assert "const-lr" in condition_label(b)


def test_pinned_lr_horizon_is_a_distinct_condition():
    """A horizon pinned past the run length is a different LR schedule."""
    a, b = cfg_for("rnd_llm", 0.5), cfg_for("rnd_llm", 0.5)
    a.schedule_timesteps = a.total_timesteps  # equivalent to leaving it unset
    b.schedule_timesteps = int(b.total_timesteps) * 4
    assert condition_label(a) != condition_label(b)
    assert condition_label(a) == condition_label(cfg_for("rnd_llm", 0.5))
