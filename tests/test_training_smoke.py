"""End-to-end smoke tests. Marked `slow` -- run with `pytest -m slow`.

These are the only tests that actually train.
"""

import csv
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

BASE = [
    sys.executable,
    "-m",
    "rl_final.ppo",  # the documented entrypoint, not ppo_minigrid -- see __main__.py
    "env=empty_5x5",
    "bonus=none",
    "eval_interval=5",
]

# The schema the analysis pipeline reads. Widening it is fine; reordering or dropping a
# column is not, because a finished run's csv cannot be regenerated without retraining.
EVAL_CSV_COLUMNS = [
    "eval_steps",
    "eval_rewards",
    "subgoals_completed",
    "entropy",
    "value_loss",
    "episodic_length",
    "beta_llm",
]


def train(tmp_path, steps=8000, seed=0, extra=()):
    out = tmp_path / "run"
    subprocess.run(
        [*BASE, f"total_timesteps={steps}", f"seed={seed}", f"hydra.run.dir={out}", *extra],
        check=True,
        capture_output=True,
        text=True,
    )
    return out


def read_csv(run_dir):
    with open(run_dir / "eval_rewards.csv") as f:
        return list(csv.DictReader(f))


def test_run_writes_every_artifact_the_analysis_needs(tmp_path):
    run_dir = train(tmp_path)
    assert (run_dir / "eval_rewards.csv").exists()
    assert (run_dir / "agent.pt").exists()
    assert list(run_dir.glob("events.out.tfevents.*")), "no TensorBoard events"
    assert (run_dir / ".hydra" / "config.yaml").exists(), "no resolved config dump"


def test_eval_csv_has_the_documented_schema(tmp_path):
    rows = read_csv(train(tmp_path))
    assert rows, "csv has no data rows"
    assert list(rows[0]) == EVAL_CSV_COLUMNS
    steps = [int(r["eval_steps"]) for r in rows]
    assert steps == sorted(steps), "eval_steps must be monotonically increasing"


def test_learning_happens_on_a_trivial_env(tmp_path):
    """Empty-5x5 is solvable in ~5 steps; a working PPO must beat a random policy.
    This is the loosest possible check that the whole loop
    -- GAE, ratio, update -- is wired correctly."""
    rows = read_csv(train(tmp_path, steps=50000))
    assert float(rows[-1]["eval_rewards"]) > 0.5


def test_same_seed_reproduces(tmp_path):
    a = read_csv(train(tmp_path / "a", seed=7))
    b = read_csv(train(tmp_path / "b", seed=7))
    assert [r["eval_rewards"] for r in a] == [r["eval_rewards"] for r in b]


def test_different_seeds_diverge(tmp_path):
    """Guards against a seed that is silently ignored -- which would make the
    3-seed error bars meaningless."""
    a = read_csv(train(tmp_path / "a", seed=0))
    b = read_csv(train(tmp_path / "b", seed=21))
    assert [r["eval_rewards"] for r in a] != [r["eval_rewards"] for r in b]


def test_capture_video_produces_mp4s(tmp_path):
    run_dir = train(tmp_path, extra=["capture_video=true"])
    videos = list(run_dir.glob("videos/step_*/*.mp4"))
    assert videos, "capture_video=true produced no mp4"
    # step-named dirs, so successive evals must not overwrite each other
    assert len({v.parent.name for v in videos}) == len(videos)
