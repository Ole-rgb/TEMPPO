import gymnasium as gym
import pytest
import torch
from minigrid.wrappers import ImgObsWrapper

from rl_final.ppo.ppo_minigrid import Agent


@pytest.fixture
def envs():
    return gym.vector.SyncVectorEnv(
        [lambda: ImgObsWrapper(gym.make("MiniGrid-Empty-5x5-v0")) for _ in range(4)]
    )


@pytest.mark.parametrize("batch", [1, 4, 8, 128])
def test_agent_is_batch_agnostic(envs, batch):
    """Nothing may bind the network to num_envs: evaluate() uses batch 1, the
    rollout uses num_envs, the update uses minibatch_size."""
    agent = Agent(envs)
    action, logprob, entropy, value = agent.get_action_and_value(torch.zeros(batch, 7, 7, 3))
    assert action.shape == (batch,)
    assert logprob.shape == (batch,)
    assert entropy.shape == (batch,)
    assert value.shape == (batch, 1)


def test_flatten_dim_matches_conv_output(envs):
    """3x conv(kernel=2, stride=1) on 7x7 -> 4x4, times 64 channels."""
    agent = Agent(envs)
    assert agent.actor.in_features == 64 * 4 * 4 == 1024
    assert agent.critic.in_features == agent.actor.in_features


def test_heads_match_the_single_env_spaces(envs):
    agent = Agent(envs)
    assert agent.actor.out_features == envs.single_action_space.n
    assert agent.critic.out_features == 1
