"""GAE correctness against hand-computable cases."""

import pytest
import torch
from rl_final.ppo.ppo_minigrid import compute_gae

GAMMA, LAM = 0.99, 0.95


def rollout(T, reward=0.0, value=0.0, boundary_at=None, terminal=False):
    """T steps, one env. `boundary_at` = index whose flags mark an episode end."""
    rewards = torch.full((T, 1), float(reward))
    values = torch.full((T, 1), float(value))
    dones = torch.zeros(T, 1)
    terminateds = torch.zeros(T, 1)
    if boundary_at is not None:
        dones[boundary_at] = 1.0
        if terminal:
            terminateds[boundary_at] = 1.0
    return dict(
        rewards=rewards,
        values=values,
        dones=dones,
        terminateds=terminateds,
        next_value=torch.full((1, 1), float(value)),
        next_done=torch.zeros(1),
        next_terminated=torch.zeros(1),
        gamma=GAMMA,
        gae_lambda=LAM,
    )


def run(**kw):
    adv, ret = compute_gae(**kw)
    return adv[:, 0], ret[:, 0]


def test_zero_reward_zero_value_gives_zero_advantage():
    adv, ret = run(**rollout(6))
    torch.testing.assert_close(adv, torch.zeros(6))
    torch.testing.assert_close(ret, torch.zeros(6))


def test_returns_equal_advantages_plus_values():
    kw = rollout(6, reward=0.3, value=1.1)
    adv, ret = run(**kw)
    torch.testing.assert_close(ret, adv + kw["values"][:, 0])


def test_single_reward_propagates_backwards_with_gamma_lambda_decay():
    """With V=0 everywhere, delta_t = r_t, so A_t = sum_k (gamma*lam)^k r_{t+k}."""
    kw = rollout(5)
    kw["rewards"][4, 0] = 1.0
    adv, _ = run(**kw)
    expected = torch.tensor([(GAMMA * LAM) ** (4 - t) for t in range(5)])
    torch.testing.assert_close(adv, expected)


def test_terminal_does_not_bootstrap():
    """Episode ends at index 3 by TERMINATION. Step 2's delta must not see V(s')."""
    kw = rollout(6, value=2.0, boundary_at=3, terminal=True)
    adv, _ = run(**kw)
    # delta_2 = r + gamma*V*nonterminal - V = 0 + 0 - 2.0
    assert adv[2].item() == pytest.approx(-2.0, abs=1e-5)


def test_truncation_does_bootstrap():
    """Same rollout, but the episode ends by TRUNCATION. Step 2 must bootstrap V(s'),
    so delta_2 = gamma*V - V, which is much less negative than the terminal case.

    This is the exact bug: CleanRL zeroes the bootstrap here too.
    """
    kw = rollout(6, value=2.0, boundary_at=3, terminal=False)
    adv, _ = run(**kw)
    assert adv[2].item() == pytest.approx(GAMMA * 2.0 - 2.0, abs=1e-5)  # -0.02, not -2.0


def test_truncation_and_termination_differ():
    """Guards against the two masks being collapsed back into one."""
    trunc, _ = run(**rollout(6, value=2.0, boundary_at=3, terminal=False))
    term, _ = run(**rollout(6, value=2.0, boundary_at=3, terminal=True))
    assert not torch.allclose(trunc, term)
    assert trunc[2] > term[2], "truncation must be valued higher than a real terminal"


def test_advantage_does_not_leak_across_an_episode_boundary():
    """A reward AFTER the boundary must not raise advantages BEFORE it, whether the
    episode terminated or was truncated."""
    for terminal in (True, False):
        kw = rollout(8, boundary_at=4, terminal=terminal)
        kw["rewards"][6, 0] = 5.0  # reward in the NEXT episode
        adv, _ = run(**kw)
        assert torch.allclose(adv[:4], torch.zeros(4)), (
            f"reward leaked backwards across the boundary (terminal={terminal})"
        )
        assert adv[6].item() == pytest.approx(5.0, abs=1e-5)


def test_seam_uses_the_carried_flags():
    """The last step of a rollout bootstraps from next_value, gated by next_terminated
    (not next_done) -- a rollout ending exactly on a truncation must still bootstrap."""
    kw = rollout(4, value=3.0)
    kw["next_done"] = torch.ones(1)  # episode ended right at the seam...
    kw["next_terminated"] = torch.zeros(1)  # ...by truncation
    adv, _ = run(**kw)
    assert adv[3].item() == pytest.approx(GAMMA * 3.0 - 3.0, abs=1e-5)

    kw["next_terminated"] = torch.ones(1)  # same seam, but a real terminal
    adv, _ = run(**kw)
    assert adv[3].item() == pytest.approx(-3.0, abs=1e-5)


def test_multiple_envs_are_independent():
    """env 0 terminates at index 3, env 1 runs on. One env's episode boundary must
    not perturb the other's advantages -- checked against a solo run of env 1."""
    T = 6
    kw = dict(
        rewards=torch.zeros(T, 2),
        values=torch.full((T, 2), 2.0),
        dones=torch.zeros(T, 2),
        terminateds=torch.zeros(T, 2),
        next_value=torch.full((1, 2), 2.0),
        next_done=torch.zeros(2),
        next_terminated=torch.zeros(2),
        gamma=GAMMA,
        gae_lambda=LAM,
    )
    kw["dones"][3, 0] = 1.0
    kw["terminateds"][3, 0] = 1.0
    adv, _ = compute_gae(**kw)

    solo, _ = run(**rollout(T, value=2.0))  # env 1 alone, no boundary
    torch.testing.assert_close(adv[:, 1], solo)  # unaffected by env 0
    assert adv[2, 0].item() == pytest.approx(-2.0, abs=1e-5)  # env 0 cut at its terminal
    assert not torch.allclose(adv[:, 0], adv[:, 1])
