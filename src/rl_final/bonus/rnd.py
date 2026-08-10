"""Random Network Distillation as a PPO exploration bonus.

The intrinsic reward is folded into the extrinsic stream before GAE:
    r_t = r_ext_t + beta_rnd * r_int_t

Everything here is gated on `cfg.bonus.beta_rnd > 0`. The `cfg.rnd.*` block holds
hyperparameters that stay FIXED across all four conditions; only `cfg.bonus.beta_rnd`
decides whether the bonus exists at all.

This is a reduced version of Burda et al. (2018). What remains is the method itself:
A frozen random target, a predictor chasing it, and the prediction error as a novelty signal.
"""

import numpy as np
import torch
import torch.nn as nn

# MiniGrid's symbolic channels have KNOWN fixed bounds -- object id 0-10, color 0-5, state 0-2.
OBS_MAX = (10.0, 5.0, 2.0)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Mirrors `ppo_minigrid.layer_init`.

    Duplicated rather than imported: `ppo_minigrid` imports `RND` at module level, so
    importing back the other way would be circular.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningMeanStd:
    """Running mean/variance of a scalar stream (Chan's parallel algorithm).

    Chunking-invariant: feeding one rollout at a time gives the same answer as one big
    batch, so the bonus scale cannot drift with `num_steps`.
    """

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        batch_mean, batch_var, batch_count = x.mean(), x.var(), x.size

        delta = batch_mean - self.mean
        total = self.count + batch_count
        m_2 = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / total
        )
        self.mean += delta * batch_count / total
        self.var = m_2 / total
        self.count = total

    @property
    def std(self):
        return float(np.sqrt(self.var))


class RND(nn.Module):
    """Novelty bonus = prediction error of a trained net against a frozen random net.

    A state visited often is easy for the predictor to match, so the error -- and the
    bonus -- decays. A state never seen keeps a large error.
    """

    def __init__(self, cfg, envs, device=None):
        super().__init__()
        h, w, c = envs.single_observation_space.shape
        assert c == len(OBS_MAX), f"expected {len(OBS_MAX)} symbolic channels, got {c}"
        embed_dim = int(cfg.rnd.embed_dim)
        hidden = int(cfg.rnd.predictor_hidden_dim)

        target_trunk, n_flat = self._trunk(c, h, w)
        self.target = nn.Sequential(target_trunk, layer_init(nn.Linear(n_flat, embed_dim)))
        # The target must stay FIXED
        for param in self.target.parameters():
            param.requires_grad_(False)

        # The predictor gets extra capacity over the target
        predictor_trunk, n_flat = self._trunk(c, h, w)
        self.predictor = nn.Sequential(
            predictor_trunk,
            layer_init(nn.Linear(n_flat, hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, embed_dim)),
        )

        # A buffer, so OBS_MAX follows .to(device).
        self.register_buffer("obs_scale", torch.tensor(OBS_MAX).view(1, c, 1, 1))
        if device is not None:
            self.to(device)

        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=float(cfg.rnd.predictor_lr)
        )

        self.normalize_intrinsic_reward = bool(cfg.rnd.normalize_intrinsic_reward)
        self.reward_rms = RunningMeanStd()

    @staticmethod
    def _trunk(c, h, w):
        """The conv stack, plus its flattened width. Same shape as the policy's."""
        net = nn.Sequential(
            layer_init(nn.Conv2d(c, 16, kernel_size=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = net(torch.zeros(1, c, h, w)).shape[1]
        return net, n_flat

    def _encode(self, obs):
        """(..., H, W, C) -> (N, C, H, W), each channel scaled into [0, 1].

        Leading dimensions are flattened, so this accepts a single step (N, H, W, C) or
        a whole rollout (T, N, H, W, C).
        """
        obs = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        return obs.permute(0, 3, 1, 2).float() / self.obs_scale

    @torch.no_grad()
    def intrinsic_reward(self, obs):
        """Per-sample novelty of `obs`: mean squared target/predictor disagreement."""
        x = self._encode(obs)
        return ((self.target(x) - self.predictor(x)) ** 2).mean(dim=-1)

    def normalize_rewards(self, int_rewards):
        """Divide the bonus by the running std, to adjust for arbitrary raw error's magnitude."""
        if not self.normalize_intrinsic_reward:
            return int_rewards
        self.reward_rms.update(int_rewards.detach().cpu().numpy())
        return int_rewards / (self.reward_rms.std + 1e-8)

    def update(self, obs):
        """One predictor gradient step over `obs`. Returns the loss."""
        x = self._encode(obs)
        with torch.no_grad():
            target = self.target(x)
        loss = ((self.predictor(x) - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
