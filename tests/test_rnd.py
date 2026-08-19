"""Tests for the RND exploration bonus and its wiring into the trainer.

The regression guards matter most: `beta_rnd = 0` must leave the `none` and `llm`
conditions exactly as they were, or the whole ablation is comparing the wrong things.
"""

import csv
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rl_final.bonus.rnd import OBS_MAX, RND, RunningMeanStd
from rl_final.ppo.ppo_minigrid import Agent, Rollout, collect_rollout, make_env

NUM_STEPS, NUM_ENVS = 16, 4
DEVICE = torch.device("cpu")


def make_cfg(**over):
    """A cfg with the same shape as configs/config.yaml's `rnd:` block."""
    rnd = dict(
        beta=0.001,
        predictor_hidden_dim=64,
        predictor_lr=1e-3,
        embed_dim=64,
        normalize_intrinsic_reward=True,
    )
    return OmegaConf.create({"rnd": {**rnd, **over}})


@pytest.fixture
def envs():
    v = gym.vector.SyncVectorEnv([make_env("MiniGrid-Empty-5x5-v0", i) for i in range(NUM_ENVS)])
    v = gym.wrappers.vector.RecordEpisodeStatistics(v)
    v.action_space.seed(0)
    return v


@pytest.fixture
def rnd(envs):
    torch.manual_seed(0)
    return RND(make_cfg(), envs)


def random_obs(n=8, seed=0):
    """Synthetic observations respecting MiniGrid's per-channel bounds (10, 5, 2)."""
    g = torch.Generator().manual_seed(seed)
    channels = [torch.randint(0, int(hi) + 1, (n, 7, 7, 1), generator=g) for hi in OBS_MAX]
    return torch.cat(channels, dim=-1).float()


# --- RunningMeanStd ---------------------------------------------------------------


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    stream = rng.normal(3.0, 2.0, size=500)

    rms = RunningMeanStd()
    for chunk in np.split(stream, 10):
        rms.update(chunk)

    assert rms.mean == pytest.approx(stream.mean(), rel=1e-6)
    assert rms.std == pytest.approx(stream.std(), rel=1e-3)


def test_running_mean_std_is_invariant_to_chunking():
    """One rollout at a time must give the same answer as one big batch -- otherwise
    the normalizer depends on num_steps and the bonus scale drifts with it."""
    rng = np.random.default_rng(1)
    stream = rng.normal(size=400)

    a, b = RunningMeanStd(), RunningMeanStd()
    a.update(stream)
    for chunk in np.split(stream, 20):
        b.update(chunk)

    assert a.mean == pytest.approx(b.mean, rel=1e-9)
    assert a.var == pytest.approx(b.var, rel=1e-9)


def test_running_mean_std_flattens_a_time_major_batch():
    """The trainer hands it (num_steps, num_envs), not a flat stream."""
    rng = np.random.default_rng(2)
    stream = rng.normal(size=(NUM_STEPS, NUM_ENVS))
    rms = RunningMeanStd()
    rms.update(stream)
    assert rms.count == pytest.approx(NUM_STEPS * NUM_ENVS, abs=1e-3)
    assert rms.mean == pytest.approx(stream.mean(), rel=1e-4)


# --- The frozen target ------------------------------------------------------------


def test_target_parameters_require_no_grad(rnd):
    assert all(not p.requires_grad for p in rnd.target.parameters())
    assert any(p.requires_grad for p in rnd.predictor.parameters())


def test_target_receives_no_gradient_and_never_moves(rnd):
    """If the target could learn it would collapse onto the predictor, the error would
    go to zero and the bonus would silently disappear."""
    before = [p.clone() for p in rnd.target.parameters()]
    rnd.update(random_obs())

    assert all(p.grad is None for p in rnd.target.parameters())
    for a, b in zip(before, rnd.target.parameters(), strict=True):
        assert torch.equal(a, b)


def test_optimizer_holds_only_predictor_params(rnd):
    """The other half of freezing the target: it must not be in the optimizer either."""
    optimized = {id(p) for group in rnd.optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in rnd.predictor.parameters()}
    assert not optimized & {id(p) for p in rnd.target.parameters()}


def test_predictor_actually_moves(rnd):
    before = [p.clone() for p in rnd.predictor.parameters()]
    rnd.update(random_obs())
    assert any(
        not torch.equal(a, b) for a, b in zip(before, rnd.predictor.parameters(), strict=True)
    )


# --- The property the method rests on ---------------------------------------------


def test_predictor_loss_falls_on_a_repeated_observation(rnd):
    obs = random_obs(n=8, seed=1)
    first = rnd.update(obs)
    for _ in range(50):
        last = rnd.update(obs)
    assert last < first, f"predictor loss did not fall ({first:.5f} -> {last:.5f})"


def test_a_novel_observation_scores_higher_than_a_familiar_one(rnd):
    """This is the whole method: novelty must be worth more than repetition."""
    familiar = random_obs(n=4, seed=2)
    novel = random_obs(n=4, seed=99)

    for _ in range(100):
        rnd.update(familiar)

    assert rnd.intrinsic_reward(novel).mean() > rnd.intrinsic_reward(familiar).mean()


def test_intrinsic_reward_is_nonnegative_and_one_scalar_per_sample(rnd):
    obs = random_obs(n=13)
    r = rnd.intrinsic_reward(obs)
    assert r.shape == (13,)
    assert (r >= 0).all(), "a squared error came out negative"


def test_intrinsic_reward_does_not_build_a_graph(rnd):
    """It is consumed inside the rollout loop; a live graph there would leak memory."""
    assert not rnd.intrinsic_reward(random_obs()).requires_grad


# --- Normalization ----------------------------------------------------------------


def test_encode_maps_every_channel_onto_the_unit_interval(rnd):
    """No running statistics and nothing to prime: MiniGrid's bounds are constants, so
    a channel at its maximum lands exactly on 1.0 from the very first observation."""
    at_max = torch.tensor(OBS_MAX).view(1, 1, 1, 3).expand(4, 7, 7, 3)
    torch.testing.assert_close(rnd._encode(at_max), torch.ones(4, 3, 7, 7))

    x = rnd._encode(random_obs(n=16))
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_encode_scales_channels_independently(rnd):
    """A shared divisor would let the object channel (0-10) dominate the prediction
    error over the state channel (0-2). Each channel gets its own bound."""
    obs = torch.ones(1, 7, 7, 3)
    x = rnd._encode(obs)[0, :, 0, 0]
    torch.testing.assert_close(x, torch.tensor([1 / 10.0, 1 / 5.0, 1 / 2.0]))


def test_encode_accepts_a_time_major_rollout(rnd):
    """The trainer passes buf.obs, which is (num_steps, num_envs, H, W, C)."""
    x = rnd._encode(torch.zeros(NUM_STEPS, NUM_ENVS, 7, 7, 3))
    assert x.shape == (NUM_STEPS * NUM_ENVS, 3, 7, 7)


def spread_around(centre, seed=0):
    """A rollout of errors with realistic relative spread (std ~ 0.3 * mean).

    Measured on doorkey_8x8, the raw error's std runs 0.4-0.7 of its mean, so a
    constant-valued stand-in would be a degenerate test: the divisor here is the STD,
    which for a constant stream is ~0 and scales nothing.
    """
    g = torch.Generator().manual_seed(seed)
    return centre * (1.0 + 0.3 * torch.randn(NUM_STEPS, NUM_ENVS, generator=g)).abs()


def test_reward_normalization_rescales_toward_unit_size(rnd):
    """Raw errors drift by orders of magnitude over training. After dividing by their
    running std the bonus stays on a stable scale, so a fixed beta keeps meaning the
    same thing at step 1k and at step 100k."""
    scaled = rnd.normalize_rewards(spread_around(1000.0))
    assert scaled.max() < 100.0, "normalization left the bonus on its raw scale"
    assert (scaled > 0).all(), "the mean must NOT be subtracted; the bonus stays positive"


def test_reward_normalization_still_lets_the_bonus_decay(rnd):
    """The running std keeps its whole history, so a shrinking raw error must still
    show up as a shrinking bonus. If it renormalized to unit size every rollout the
    bonus would never decay and RND would degenerate into a constant survival reward.
    """
    first = rnd.normalize_rewards(spread_around(10.0, seed=1)).mean()
    for i in range(5):
        last = rnd.normalize_rewards(spread_around(0.1, seed=2 + i)).mean()
    assert last < first / 10


def test_reward_normalization_is_a_no_op_when_disabled(envs):
    torch.manual_seed(0)
    off = RND(make_cfg(normalize_intrinsic_reward=False), envs)
    raw = torch.rand(NUM_STEPS, NUM_ENVS)
    torch.testing.assert_close(off.normalize_rewards(raw), raw)


# --- Wiring into the rollout ------------------------------------------------------


def fresh_state(envs):
    obs, _ = envs.reset(seed=0)
    return torch.Tensor(obs), torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS)


def test_int_rewards_stay_zero_without_rnd(envs):
    torch.manual_seed(0)
    agent = Agent(envs)
    buf = Rollout.empty(NUM_STEPS, NUM_ENVS, envs, DEVICE)
    collect_rollout(agent, envs, buf, *fresh_state(envs), 0, DEVICE)
    assert torch.count_nonzero(buf.int_rewards) == 0


def test_rnd_fills_int_rewards_without_perturbing_the_policy(envs):
    """RND must be a pure observer during collection: same seed, same actions, same
    extrinsic rewards, with or without it. If it consumed randomness the `rnd` and
    `none` conditions would diverge for a reason that has nothing to do with the bonus.
    """
    torch.manual_seed(0)
    agent = Agent(envs)
    rnd = RND(make_cfg(), envs)

    plain = Rollout.empty(NUM_STEPS, NUM_ENVS, envs, DEVICE)
    torch.manual_seed(123)
    collect_rollout(agent, envs, plain, *fresh_state(envs), 0, DEVICE)

    bonused = Rollout.empty(NUM_STEPS, NUM_ENVS, envs, DEVICE)
    torch.manual_seed(123)
    collect_rollout(agent, envs, bonused, *fresh_state(envs), 0, DEVICE, rnd)

    torch.testing.assert_close(plain.actions, bonused.actions)
    torch.testing.assert_close(plain.rewards, bonused.rewards)
    torch.testing.assert_close(plain.obs, bonused.obs)
    assert torch.count_nonzero(bonused.int_rewards) > 0


# --- End to end -------------------------------------------------------------------

BASE = [sys.executable, "-m", "rl_final.ppo", "env=empty_5x5", "eval_interval=5"]

# Kept in sync with tests/test_training_smoke.py: an RND run must write the same
# columns as a vanilla one, so the two are directly loadable side by side.
EVAL_CSV_COLUMNS = [
    "eval_steps",
    "eval_rewards",
    "subgoals_completed",
    "entropy",
    "value_loss",
    "episodic_length",
    "beta_llm",
]


def train(out, seed=0, steps=8000, extra=()):
    subprocess.run(
        [*BASE, f"total_timesteps={steps}", f"seed={seed}", f"hydra.run.dir={out}", *extra],
        check=True,
        capture_output=True,
        text=True,
    )
    return out


def eval_returns(run_dir):
    with open(run_dir / "eval_rewards.csv") as f:
        return [r["eval_rewards"] for r in csv.DictReader(f)]


@pytest.mark.slow
def test_beta_rnd_zero_reproduces_vanilla_ppo(tmp_path):
    """THE regression guard. `bonus=rnd` with beta driven to zero must take exactly the
    vanilla code path, which protects the `none` and `llm` conditions from every change
    in this module."""
    a = train(tmp_path / "none", extra=["bonus=none"])
    b = train(tmp_path / "zero", extra=["bonus=rnd", "rnd.beta=0.0"])
    assert eval_returns(a) == eval_returns(b)


@pytest.mark.slow
def test_the_bonus_actually_changes_training(tmp_path):
    """The complement: with beta > 0 the run must differ. Catches a bonus that is
    computed, logged, and then never reaches the reward stream."""
    a = train(tmp_path / "none", extra=["bonus=none"])
    b = train(tmp_path / "rnd", extra=["bonus=rnd"])
    assert eval_returns(a) != eval_returns(b)


@pytest.mark.slow
def test_rnd_run_writes_the_bonus_diagnostics(tmp_path):
    """Without these scalars the 'intrinsic reward over training' figure cannot be made."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = train(tmp_path / "rnd", extra=["bonus=rnd"])
    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    assert {
        "charts/intrinsic_reward_mean",
        "charts/intrinsic_reward_std",
        "losses/rnd_predictor_loss",
    } <= tags

    # And the eval CSV schema the whole analysis pipeline depends on is unchanged.
    with open(run_dir / "eval_rewards.csv") as f:
        assert list(next(csv.DictReader(f))) == EVAL_CSV_COLUMNS
