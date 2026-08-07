import gymnasium as gym
import numpy as np
import pytest
from rl_final.ppo.ppo_minigrid import EVAL_SEED_BASE, make_env

ENV_IDS = ["MiniGrid-Empty-5x5-v0", "MiniGrid-DoorKey-8x8-v0", "MiniGrid-MultiRoom-N4-S5-v1"]


def test_make_env_returns_a_thunk_not_an_env():
    """SyncVectorEnv needs constructors; evaluate() calls the thunk immediately."""
    thunk = make_env("MiniGrid-Empty-5x5-v0", 0)
    assert callable(thunk)
    assert isinstance(thunk(), gym.Env)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_obs_is_the_bare_image(env_id):
    """ImgObsWrapper must strip the {image, direction, mission} dict."""
    env = make_env(env_id, 0)()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (7, 7, 3)
    assert obs.dtype == np.uint8


def test_vector_env_space_distinction():
    """Agent is built from single_* spaces; using the batched ones would bake in num_envs."""
    envs = gym.vector.SyncVectorEnv([make_env("MiniGrid-Empty-5x5-v0", i) for i in range(4)])
    assert envs.single_observation_space.shape == (7, 7, 3)
    assert envs.observation_space.shape == (4, 7, 7, 3)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)


def test_truncation_returns_the_final_obs_not_the_reset_obs():
    """Gymnasium's NEXT_STEP autoreset is why V(s_final) is already values[t+1].
    If this ever flips to SAME_STEP, the GAE bootstrap silently reads the wrong state.
    """
    envs = gym.vector.SyncVectorEnv([make_env("MiniGrid-Empty-5x5-v0", 0)])
    fresh, _ = envs.reset(seed=0)
    for _ in range(200):
        obs, _, te, tr, _ = envs.step(np.array([2]))
        if te[0] or tr[0]:
            assert not np.array_equal(obs, fresh), "reset happened on the terminating step"
            after, *_ = envs.step(np.array([2]))
            assert np.array_equal(after, fresh), "reset did not happen on the following step"
            return
    pytest.fail("episode never ended")


def test_eval_seeds_are_deterministic():
    """evaluate() must score every seed and condition on identical episodes."""
    a = make_env("MiniGrid-DoorKey-8x8-v0", 0)()
    b = make_env("MiniGrid-DoorKey-8x8-v0", 0)()
    for ep in range(3):
        o1, _ = a.reset(seed=EVAL_SEED_BASE + ep)
        o2, _ = b.reset(seed=EVAL_SEED_BASE + ep)
        assert np.array_equal(o1, o2)
