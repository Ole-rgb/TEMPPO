"""Tests for the LLM subgoal bonus, focused on `skip_go_to_goal`.

`go_to_goal` fires on `terminated`, the same event the env already rewards, so dropping
it changes what the shaped MDP looks like. That makes it an experimental condition, not
a cosmetic flag: if it silently stopped taking effect, a with/without ablation would
compare two identical settings and report a null result. Hence the guards here.

The cache is built from scratch against a real layout rather than read off disk, so the
tests do not depend on which plans happen to have been bought.
"""

import gymnasium as gym
import minigrid  # noqa: F401  -- registers the MiniGrid env ids
import pytest

from rl_final.bonus.llm import (
    Subgoal,
    SubgoalCache,
    SubgoalTracker,
    anneal_factor,
    plan_key_and_roles,
)

ENV_ID = "MiniGrid-DoorKey-8x8-v0"
BONUS = 1.0


@pytest.fixture
def envs():
    v = gym.vector.SyncVectorEnv([lambda: gym.make(ENV_ID)])
    v.reset(seed=0)
    yield v
    v.close()


@pytest.fixture
def cache(tmp_path, envs):
    """A one-entry cache holding `open_door` then `go_to_goal` for this layout."""
    base = envs.unwrapped.envs[0].unwrapped
    key, roles = plan_key_and_roles(base)
    colour = next(iter(roles))  # a colour this layout actually drew
    c = SubgoalCache(tmp_path / "cache.jsonl")
    c.add(
        key,
        base.mission,
        "",
        [Subgoal(action="open_door", color=colour), Subgoal(action="go_to_goal")],
        roles,
    )
    return c


def actions(tracker, i=0):
    return [s.action for s in tracker.plans[i]]


def test_terminal_subgoal_dropped_only_when_skipping(envs, cache):
    keep = SubgoalTracker(envs, cache, BONUS, None, skip_go_to_goal=False)
    drop = SubgoalTracker(envs, cache, BONUS, None, skip_go_to_goal=True)

    assert actions(keep) == ["open_door", "go_to_goal"]
    assert actions(drop) == ["open_door"]


def test_reaching_the_goal_pays_only_when_the_terminal_subgoal_is_kept(envs, cache):
    """The behaviour that matters: same layout, same plan, different reward on success."""
    paid = {}
    for skip in (False, True):
        t = SubgoalTracker(envs, cache, BONUS, None, skip_go_to_goal=skip)
        t.idx[0] = len(t.plans[0])  # the door is already credited; only the goal is left
        if not skip:
            t.idx[0] -= 1
        paid[skip] = float(t.step(dones=[True], terminateds=[True])[0])

    assert paid[False] == pytest.approx(BONUS)
    assert paid[True] == 0.0


def test_filtering_is_not_counted_as_a_cache_miss(envs, cache):
    """An empty plan from filtering is a design choice; a miss is a missing plan."""
    t = SubgoalTracker(envs, cache, BONUS, None, skip_go_to_goal=True)
    assert t.misses == 0


def test_a_plan_absent_from_the_cache_still_misses(envs, tmp_path):
    """The miss path has to survive the filter being on."""
    t = SubgoalTracker(envs, SubgoalCache(tmp_path / "empty.jsonl"), BONUS, None)
    assert t.plans[0] == []
    assert t.misses == 1


# --- warm-start schedule ----------------------------------------------------------


def test_no_annealing_holds_beta_flat():
    """`anneal_steps=0` is what every constant-beta condition passes; it must be inert."""
    assert [anneal_factor(t, 0) for t in (0, 10_000, 750_000)] == [1.0, 1.0, 1.0]
    assert anneal_factor(750_000, -1) == 1.0


def test_annealing_decays_linearly_then_pins_at_zero():
    assert anneal_factor(0, 150_000) == pytest.approx(1.0)
    assert anneal_factor(75_000, 150_000) == pytest.approx(0.5)
    assert anneal_factor(150_000, 150_000) == pytest.approx(0.0)
    # Past the horizon the condition IS PPO+RND -- never a negative bonus.
    assert anneal_factor(600_000, 150_000) == 0.0


def test_horizon_is_monotone_in_the_schedule():
    """A longer horizon must hold more bonus at every step, or the sweep is unordered."""
    for t in (10_000, 100_000, 149_000):
        assert anneal_factor(t, 300_000) > anneal_factor(t, 150_000)


# --- shuffled-plan control --------------------------------------------------------
#
# The arm claims to isolate the LLM's ORDERING from its choice of targets. That only
# holds if permuting leaves the multiset of subgoals untouched -- if a future edit made
# shuffling also drop or duplicate entries, the arm would quietly start measuring "fewer
# subgoals" and still look like a working control.


@pytest.fixture
def long_cache(tmp_path, envs):
    """A four-step plan, long enough that a permutation is not usually the identity."""
    base = envs.unwrapped.envs[0].unwrapped
    key, roles = plan_key_and_roles(base)
    colour = next(iter(roles))
    c = SubgoalCache(tmp_path / "long.jsonl")
    c.add(
        key,
        base.mission,
        "",
        [
            Subgoal(action="pick_up", color=colour, object="key"),
            Subgoal(action="open_door", color=colour),
            Subgoal(action="pick_up", color=colour, object="ball"),
            Subgoal(action="go_to_goal"),
        ],
        roles,
    )
    return c


def track(envs, cache, **kw):
    return SubgoalTracker(envs, cache, BONUS, None, skip_go_to_goal=False, **kw)


def test_shuffling_is_off_by_default(envs, long_cache):
    assert actions(track(envs, long_cache)) == [
        "pick_up",
        "open_door",
        "pick_up",
        "go_to_goal",
    ]


def test_shuffling_preserves_the_multiset_of_subgoals(envs, long_cache):
    """The whole point of the control: same subgoals, same total bonus, new order."""
    ordered = track(envs, long_cache).plans[0]
    for seed in range(8):
        shuffled = track(envs, long_cache, shuffle_plan=True, seed=seed).plans[0]
        assert sorted(s.model_dump_json() for s in shuffled) == sorted(
            s.model_dump_json() for s in ordered
        )


def test_shuffling_is_reproducible_for_a_seed(envs, long_cache):
    a = actions(track(envs, long_cache, shuffle_plan=True, seed=7))
    b = actions(track(envs, long_cache, shuffle_plan=True, seed=7))
    assert a == b


def test_shuffling_actually_reorders_for_some_seed(envs, long_cache):
    """Guards the opposite failure: a no-op permutation would pass every check above."""
    ordered = actions(track(envs, long_cache))
    assert any(
        actions(track(envs, long_cache, shuffle_plan=True, seed=s)) != ordered for s in range(8)
    )


def test_shuffling_does_not_touch_the_global_rng(envs, long_cache):
    """np.random also drives PPO's minibatch shuffling, so building the tracker must
    not consume from it -- otherwise enabling the LLM bonus perturbs the optimizer."""
    import numpy as np

    np.random.seed(0)
    before = np.random.get_state()[2]
    track(envs, long_cache, shuffle_plan=True, seed=3)
    assert np.random.get_state()[2] == before
