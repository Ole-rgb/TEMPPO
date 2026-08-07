"""Tests for the extracted rollout collection and PPO update."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from rl_final.ppo.ppo_minigrid import Agent, Rollout, collect_rollout, make_env, ppo_update

NUM_STEPS, NUM_ENVS = 32, 4
DEVICE = torch.device("cpu")


@pytest.fixture
def envs():
    v = gym.vector.SyncVectorEnv([make_env("MiniGrid-Empty-5x5-v0", i) for i in range(NUM_ENVS)])
    v = gym.wrappers.vector.RecordEpisodeStatistics(v)
    v.action_space.seed(0)
    return v


@pytest.fixture
def agent(envs):
    torch.manual_seed(0)
    return Agent(envs)


def fresh_state(envs):
    obs, _ = envs.reset(seed=0)
    return (
        torch.Tensor(obs),
        torch.zeros(NUM_ENVS),
        torch.zeros(NUM_ENVS),
    )


# --- Rollout buffer ---------------------------------------------------------------


def test_flatten_is_row_major_and_keeps_rows_together(envs):
    """If flatten transposed or mismatched, advantages would be paired with the wrong
    observations and PPO would silently train on garbage."""
    buf = Rollout.empty(NUM_STEPS, NUM_ENVS, envs, DEVICE)
    buf.obs.copy_(torch.randn_like(buf.obs))
    buf.values.copy_(torch.randn_like(buf.values))
    advantages = torch.randn(NUM_STEPS, NUM_ENVS)
    returns = torch.randn(NUM_STEPS, NUM_ENVS)

    batch = buf.flatten(advantages, returns)
    assert batch.obs.shape == (NUM_STEPS * NUM_ENVS, 7, 7, 3)
    assert batch.advantages.shape == (NUM_STEPS * NUM_ENVS,)
    for t in (3, 17, 31):
        for e in range(NUM_ENVS):
            i = t * NUM_ENVS + e
            torch.testing.assert_close(batch.obs[i], buf.obs[t, e])
            torch.testing.assert_close(batch.advantages[i], advantages[t, e])
            torch.testing.assert_close(batch.values[i], buf.values[t, e])


# --- collect_rollout --------------------------------------------------------------


def test_global_step_counts_env_frames(envs, agent):
    buf = Rollout.empty(NUM_STEPS, NUM_ENVS, envs, DEVICE)
    obs, done, term = fresh_state(envs)
    *_, global_step, _ = collect_rollout(agent, envs, buf, obs, done, term, 1000, DEVICE)
    assert global_step == 1000 + NUM_STEPS * NUM_ENVS


def test_collect_rollout_aligns_dones_and_terminateds(envs, agent):
    """REGRESSION: dones[t] and terminateds[t] must describe the same timestep.

    `done = terminated or truncated`, so terminated=1 with done=0 is impossible when
    aligned -- and is exactly what the off-by-one produced.
    """
    buf = Rollout.empty(NUM_STEPS, NUM_ENVS, envs, DEVICE)
    obs, done, term = fresh_state(envs)
    saw_termination = False

    for _ in range(40):  # enough rollouts to see real terminations
        obs, done, term, _, _ = collect_rollout(agent, envs, buf, obs, done, term, 0, DEVICE)
        assert torch.all(buf.terminateds <= buf.dones), "terminated=1 with done=0"
        saw_termination |= bool(buf.terminateds.any())

    assert saw_termination, "no episode ever terminated; the invariant was never exercised"


def test_episode_stats_are_reported_for_finished_episodes(envs, agent):
    buf = Rollout.empty(NUM_STEPS, NUM_ENVS, envs, DEVICE)
    obs, done, term = fresh_state(envs)
    collected = []
    for _ in range(40):
        obs, done, term, _, stats = collect_rollout(agent, envs, buf, obs, done, term, 0, DEVICE)
        collected += stats
    assert collected, "episodes finished but none were reported"
    for _gs, ep_return, ep_length in collected:
        assert ep_length > 0
        assert 0.0 <= float(ep_return) <= 1.0  # MiniGrid reward is 1 - 0.9*steps/max_steps


# --- ppo_update -------------------------------------------------------------------


def make_args(**over):
    base = dict(
        batch_size=NUM_STEPS * NUM_ENVS,
        minibatch_size=NUM_STEPS * NUM_ENVS,
        update_epochs=1,
        norm_adv=True,
        clip_coef=0.2,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
    )
    return SimpleNamespace(**{**base, **over})


def make_batch(envs, agent):
    buf = Rollout.empty(NUM_STEPS, NUM_ENVS, envs, DEVICE)
    obs, done, term = fresh_state(envs)
    collect_rollout(agent, envs, buf, obs, done, term, 0, DEVICE)
    advantages = torch.randn(NUM_STEPS, NUM_ENVS)
    return buf.flatten(advantages, advantages + buf.values)


def test_first_update_has_ratio_one(envs, agent):
    """Diagnostics are recorded BEFORE the gradient step, so on a single-minibatch,
    single-epoch update the policy has not moved yet: ratio == 1 exactly."""
    opt = torch.optim.Adam(agent.parameters(), lr=1e-4)
    m = ppo_update(agent, opt, make_batch(envs, agent), make_args())
    assert m["losses/clipfrac"] == 0.0
    assert m["losses/approx_kl"] == pytest.approx(0.0, abs=1e-7)


def test_update_changes_the_parameters(envs, agent):
    """Directly catches a missing optimizer.step()."""
    before = [p.clone() for p in agent.parameters()]
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    ppo_update(agent, opt, make_batch(envs, agent), make_args())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.parameters(), strict=True))


def offset_batch(envs, agent, log_offset, advantage):
    """A batch whose stored logprobs deliberately DISAGREE with the current policy, so
    ratio = exp(-log_offset) != 1 on the very first minibatch.

    Without this every update test sits at ratio==1, where clipping never engages and
    the sign of logratio is invisible -- so those bugs go undetected.
    """
    batch = make_batch(envs, agent)
    with torch.no_grad():
        _, logp, _, _ = agent.get_action_and_value(batch.obs, batch.actions.long())
    batch.logprobs = logp - log_offset
    batch.advantages = torch.full_like(batch.advantages, float(advantage))
    return batch


def moved(agent, fn):
    before = [p.clone() for p in agent.parameters()]
    fn()
    return sum((a - b).abs().sum().item() for a, b in zip(before, agent.parameters(), strict=True))


def test_positive_advantage_raises_the_logprob_of_the_taken_actions(envs, agent):
    """Sign check on `logratio = newlogprob - b_logprobs`. With a positive advantage the
    update must make the stored actions MORE likely. A flipped subtraction inverts the
    ratio and drives them less likely, with no crash."""
    batch = offset_batch(envs, agent, log_offset=0.1, advantage=1.0)
    args = make_args(norm_adv=False, ent_coef=0.0, vf_coef=0.0)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)

    with torch.no_grad():
        before = agent.get_action_and_value(batch.obs, batch.actions.long())[1].mean().item()
    ppo_update(agent, opt, batch, args)
    with torch.no_grad():
        after = agent.get_action_and_value(batch.obs, batch.actions.long())[1].mean().item()
    assert after > before, f"logprob fell {before:.4f} -> {after:.4f}; logratio sign inverted"


def test_clipping_suppresses_the_update_when_the_ratio_is_far_outside_the_range(envs, agent):
    """This is what makes PPO *proximal*. With ratio ~ e^2 and a positive advantage the
    clipped branch wins, is constant in theta, and contributes no gradient. Remove the
    clamp and the same batch produces a much larger step."""
    args_clipped = make_args(norm_adv=False, ent_coef=0.0, vf_coef=0.0, clip_coef=0.2)
    args_unclipped = make_args(norm_adv=False, ent_coef=0.0, vf_coef=0.0, clip_coef=1e6)

    torch.manual_seed(0)
    a1 = Agent(envs)
    d_clipped = moved(
        a1,
        lambda: ppo_update(
            a1,
            torch.optim.SGD(a1.parameters(), lr=1e-2),
            offset_batch(envs, a1, log_offset=2.0, advantage=1.0),
            args_clipped,
        ),
    )

    torch.manual_seed(0)
    a2 = Agent(envs)
    d_unclipped = moved(
        a2,
        lambda: ppo_update(
            a2,
            torch.optim.SGD(a2.parameters(), lr=1e-2),
            offset_batch(envs, a2, log_offset=2.0, advantage=1.0),
            args_unclipped,
        ),
    )

    assert d_clipped < 0.5 * d_unclipped, (
        f"clipping barely restrained the update ({d_clipped:.4f} vs {d_unclipped:.4f}); "
        "the clamp is not doing anything"
    )


def test_norm_adv_standardizes_away_a_constant_advantage(envs, agent):
    """With norm_adv on, a constant advantage normalizes to ~0 and the policy loss
    vanishes. With it off, the same batch produces a real step."""
    batch_kw = dict(log_offset=0.1, advantage=3.0)
    torch.manual_seed(0)
    a1 = Agent(envs)
    d_norm = moved(
        a1,
        lambda: ppo_update(
            a1,
            torch.optim.SGD(a1.parameters(), lr=1e-2),
            offset_batch(envs, a1, **batch_kw),
            make_args(norm_adv=True, ent_coef=0.0, vf_coef=0.0),
        ),
    )

    torch.manual_seed(0)
    a2 = Agent(envs)
    d_raw = moved(
        a2,
        lambda: ppo_update(
            a2,
            torch.optim.SGD(a2.parameters(), lr=1e-2),
            offset_batch(envs, a2, **batch_kw),
            make_args(norm_adv=False, ent_coef=0.0, vf_coef=0.0),
        ),
    )

    assert d_norm < 0.1 * d_raw, f"norm_adv had no effect ({d_norm:.4f} vs {d_raw:.4f})"


def test_entropy_is_a_bonus_not_a_penalty(envs, agent):
    """Sign check on `loss = pg - ent_coef*H + vf*v`.

    A freshly initialised agent already sits at max entropy log(7)=1.9459, so it must
    first be made peaked -- otherwise there is no headroom and the test passes/fails
    for the wrong reason. With advantages zeroed and a large ent_coef, the update must
    push entropy back UP. A flipped sign would drive it further down.
    """
    batch = make_batch(envs, agent)
    batch.advantages.zero_()
    with torch.no_grad():  # make the policy strongly prefer action 0
        agent.actor.bias[0] += 5.0
    before = agent.get_action_and_value(batch.obs)[2].mean().item()
    assert before < 1.5, "setup failed: policy is not peaked, so there is no headroom"

    opt = torch.optim.Adam(agent.parameters(), lr=1e-2)
    ppo_update(agent, opt, batch, make_args(ent_coef=10.0, vf_coef=0.0, update_epochs=4))
    after = agent.get_action_and_value(batch.obs)[2].mean().item()
    assert after > before, f"entropy fell {before:.4f} -> {after:.4f}; sign is flipped"
