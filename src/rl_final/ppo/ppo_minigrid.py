"""
ppo_minigrid.py

Adapted from CleanRL's ppo_atari.py:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py

Original license: MIT (see LICENSE file).
Modifications: replaced Atari CNN with MiniGrid encoder;
added exploration bonuses (RND, llm-subgoals).
"""

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

# from dataclasses import dataclass
import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from minigrid.wrappers import ImgObsWrapper
from omegaconf import DictConfig, OmegaConf

# import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from rl_final.bonus.llm import LLMBonus, anneal_factor
from rl_final.bonus.rnd import RND


def make_env(env_id, idx, video_dir=None):
    """video_dir: absolute path to record into, or None for no recording.

    Only episode 0 is recorded. Callers pass a *step-specific* directory so
    successive evals don't overwrite each other's rl-video-episode-0.mp4.
    """

    def thunk():
        if video_dir is not None and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, str(video_dir), episode_trigger=lambda ep: ep == 0)
        else:
            env = gym.make(env_id)
        return ImgObsWrapper(env)

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # MiniGrid's ImgObsWrapper yields HWC uint8, e.g. (7, 7, 3) for the default
        # 7x7 egocentric view. The three channels are symbolic indices
        # (object, color, state), not pixels.
        h, w, c = envs.single_observation_space.shape
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(c, 16, kernel_size=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.network(torch.zeros(1, c, h, w)).shape[1]
        self.actor = layer_init(nn.Linear(n_flat, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(n_flat, 1), std=1)

    def get_value(self, x):
        hidden = self.network(self._encode(x))
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(self._encode(x))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def _encode(self, x):
        return x.permute(0, 3, 1, 2) / 10.0  # NHWC -> NCHW


@dataclass
class Rollout:
    """The on-policy buffer: one fixed-size window of experience, shape (num_steps, num_envs).

    Refilled every iteration and discarded after the update -- PPO's importance ratio
    is only valid for data collected under the policy that produced it.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    int_rewards: torch.Tensor  # Raw RND prediction error
    llm_rewards: torch.Tensor  # subgoal_bonus per LLM subgoal completed this step
    dones: torch.Tensor
    terminateds: torch.Tensor
    values: torch.Tensor

    @classmethod
    def empty(cls, num_steps, num_envs, envs, device):
        def zeros(trailing=()):
            return torch.zeros((num_steps, num_envs) + tuple(trailing)).to(device)

        return cls(
            obs=zeros(envs.single_observation_space.shape),
            actions=zeros(envs.single_action_space.shape),
            logprobs=zeros(),
            rewards=zeros(),
            int_rewards=zeros(),
            llm_rewards=zeros(),
            dones=zeros(),
            terminateds=zeros(),
            values=zeros(),
        )

    def flatten(self, advantages, returns):
        """(num_steps, num_envs, ...) -> (batch_size, ...).

        Time ordering is destroyed here, which is fine: GAE has already run and the
        PPO loss is a sum over independent transitions.
        """
        return Batch(
            obs=self.obs.reshape((-1,) + self.obs.shape[2:]),
            actions=self.actions.reshape((-1,) + self.actions.shape[2:]),
            logprobs=self.logprobs.reshape(-1),
            values=self.values.reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
        )


@dataclass
class Batch:
    """A flattened Rollout, ready for minibatching."""

    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


def collect_rollout(
    agent,
    envs,
    buf,
    next_obs,
    next_done,
    next_terminated,
    global_step,
    device,
    rnd=None,
    llm=None,
):
    """Fill `buf` with one rollout. The policy is frozen throughout (no_grad), which is
    what makes the stored logprobs a valid `pi_old` for the whole update.

    Returns the carried state plus the episodes that finished, as
    (global_step, episodic_return, episodic_length) triples for the caller to log.

    `rnd=None` / `llm=None` are the vanilla paths: the matching buffer stays at zero and
    not a single bonus op runs, so the beta=0 conditions are bit-for-bit unaffected.
    """
    num_steps, num_envs = buf.rewards.shape
    episode_stats = []

    for step in range(num_steps):
        global_step += num_envs
        buf.obs[step] = next_obs
        buf.dones[step] = next_done
        buf.terminateds[step] = next_terminated

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            buf.values[step] = value.flatten()
        buf.actions[step] = action
        buf.logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        buf.rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done, next_terminated = (
            torch.Tensor(next_obs).to(device),
            torch.Tensor(np.logical_or(terminations, truncations)).to(device),
            torch.Tensor(terminations).to(device),
        )
        if rnd is not None:
            buf.int_rewards[step] = rnd.intrinsic_reward(next_obs)
        if llm is not None:
            buf.llm_rewards[step] = torch.as_tensor(
                llm.step(np.logical_or(terminations, truncations), terminations),
                dtype=torch.float32,
                device=device,
            )
        if "episode" in infos:
            mask = infos["_episode"]
            rs, ls = infos["episode"]["r"][mask], infos["episode"]["l"][mask]
            episode_stats += [(global_step, r, ln) for r, ln in zip(rs, ls, strict=False)]

    return next_obs, next_done, next_terminated, global_step, episode_stats


def compute_gae(
    rewards,
    values,
    dones,
    terminateds,
    next_value,
    next_done,
    next_terminated,
    gamma,
    gae_lambda,
):
    """
    Generalized Advantage Estimation over one rollout. Returns: (advantages, returns)
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nonboundary = 1.0 - next_done
            nonterminal = 1.0 - next_terminated
            nextvalues = next_value
        else:
            nonboundary = 1.0 - dones[t + 1]
            nonterminal = 1.0 - terminateds[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nonboundary * lastgaelam
    return advantages, advantages + values


def ppo_update(agent, optimizer, batch, args):
    """`update_epochs` passes over the batch, each split into `num_minibatches`.

    Mutates the agent in place; returns the diagnostics that get logged. Note the
    returned losses are from the LAST minibatch of the last epoch, not averages --
    only clipfrac is accumulated across all of them.

    Knows nothing about RND: the bonus is already inside `batch.advantages` by the time
    it gets here, and the predictor is trained separately by the caller.
    """
    b_inds = np.arange(args.batch_size)
    clipfracs = []

    for _epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            # Recompute under the CURRENT parameters. b_logprobs is frozen pi_old.
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                batch.obs[mb_inds], batch.actions.long()[mb_inds]
            )
            logratio = newlogprob - batch.logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = batch.advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            # Policy loss. min() of the objective == max() of the negated loss.
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - batch.returns[mb_inds]) ** 2
                v_clipped = batch.values[mb_inds] + torch.clamp(
                    newvalue - batch.values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - batch.returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - batch.returns[mb_inds]) ** 2).mean()

            # entropy is a BONUS, hence the minus: minimizing -H maximizes exploration.
            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    y_pred, y_true = batch.values.cpu().numpy(), batch.returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return {
        "losses/policy_loss": pg_loss.item(),
        "losses/value_loss": v_loss.item(),
        "losses/entropy": entropy_loss.item(),
        "losses/old_approx_kl": old_approx_kl.item(),
        "losses/approx_kl": approx_kl.item(),
        "losses/clipfrac": float(np.mean(clipfracs)),
        "losses/explained_variance": explained_var,
    }


# Eval layouts are drawn from a fixed seed range that is disjoint from the training
# seeds, so every training seed and every condition is scored on the same episodes.
EVAL_SEED_BASE = 10_000


def evaluate(agent, env_id, num_episodes, device, video_dir=None):
    """Mean episodic return of the current policy on a fresh, single env.

    Actions are sampled rather than taken greedily: an argmax policy in MiniGrid can
    deadlock (e.g. turning into a wall forever) and would report 0 for a policy that
    is in fact learning.

    Passing video_dir records episode 0 of this eval to an mp4.
    """
    env = make_env(env_id, 0, video_dir)()
    returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + ep)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _, _ = agent.get_action_and_value(obs_t)
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns))


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    args = cfg
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join(
                [
                    f"|{key}|{value}|"
                    for key, value in OmegaConf.to_container(cfg, resolve=True).items()
                ]
            )
        ),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i) for i in range(args.num_envs)],
    )
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), (
        "only discrete action space is supported"
    )

    envs.action_space.seed(args.seed)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # The BONUS GROUP decides which bonuses exist -- never cfg.rnd.beta / cfg.llm.beta
    beta_rnd = float(cfg.bonus.beta_rnd)
    beta_llm = float(cfg.bonus.beta_llm)
    # Warm start: 0 (the default for every non-annealed condition) holds beta_llm flat.
    anneal_llm_steps = int(cfg.bonus.get("anneal_llm_steps", 0))
    rnd = None
    if beta_rnd > 0:
        # Built after seeding, so each seed gets its own reproducible random target.
        rnd = RND(cfg, envs, device)

    eval_file = open(run_dir / "eval_rewards.csv", "w", newline="")
    eval_csv = csv.writer(eval_file)
    eval_csv.writerow(["eval_steps", "eval_rewards"])

    def run_eval():
        # One mp4 per eval, in a step-named folder, so you can watch the policy
        video_dir = run_dir / "videos" / f"step_{global_step:08d}" if args.capture_video else None
        mean_return = evaluate(agent, args.env_id, args.eval_episodes, device, video_dir)
        eval_csv.writerow([global_step, mean_return])
        eval_file.flush()
        writer.add_scalar("charts/eval_return", mean_return, global_step)
        return mean_return

    # ALGO Logic: Storage setup
    buf = Rollout.empty(args.num_steps, args.num_envs, envs, device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_terminated = torch.zeros(args.num_envs).to(device)

    llm = LLMBonus(cfg, envs) if beta_llm > 0 else None

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        next_obs, next_done, next_terminated, global_step, episode_stats = collect_rollout(
            agent, envs, buf, next_obs, next_done, next_terminated, global_step, device, rnd, llm
        )
        for gs, ep_return, ep_length in episode_stats:
            writer.add_scalar("charts/episodic_return", ep_return, gs)
            writer.add_scalar("charts/episodic_length", ep_length, gs)

        rewards = buf.rewards
        if rnd is not None:
            scaled_int_rewards = rnd.normalize_rewards(buf.int_rewards)
            rewards = rewards + beta_rnd * scaled_int_rewards
            rnd_loss = rnd.update(buf.obs)
        if llm is not None:
            beta_llm_now = beta_llm * anneal_factor(global_step, anneal_llm_steps)
            rewards = rewards + beta_llm_now * buf.llm_rewards

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages, returns = compute_gae(
                rewards,
                buf.values,
                buf.dones,
                buf.terminateds,
                next_value,
                next_done,
                next_terminated,
                args.gamma,
                args.gae_lambda,
            )

        metrics = ppo_update(agent, optimizer, buf.flatten(advantages, returns), args)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        if rnd is not None:
            # The raw mean must DECAY as states stop being novel
            writer.add_scalar(
                "charts/intrinsic_reward_mean", buf.int_rewards.mean().item(), global_step
            )
            writer.add_scalar(
                "charts/intrinsic_reward_std", buf.int_rewards.std().item(), global_step
            )
            # compare against episodic_return to see how much the bonus dominates.
            writer.add_scalar(
                "charts/intrinsic_reward_scaled_mean",
                (beta_rnd * scaled_int_rewards).mean().item(),
                global_step,
            )
            writer.add_scalar("losses/rnd_predictor_loss", rnd_loss, global_step)
        if llm is not None:
            # The warm-start schedule itself: flat at beta_llm unless it is annealing.
            writer.add_scalar("charts/beta_llm", beta_llm_now, global_step)
            # Compare against episodic_return the same way as the RND bonus.
            writer.add_scalar(
                "charts/llm_bonus_scaled_mean",
                (beta_llm_now * buf.llm_rewards).mean().item(),
                global_step,
            )
            for tag, val in llm.stats.items():
                writer.add_scalar(tag, val, global_step)
        for tag, val in metrics.items():
            writer.add_scalar(tag, val, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        print(
            f"iter {iteration:4d}/{args.num_iterations}  "
            f"steps {global_step:>8,}  "
            f"ent {metrics['losses/entropy']:5.3f}  kl {metrics['losses/approx_kl']:.4f}  "
            f"clip {metrics['losses/clipfrac']:.3f}  ev {metrics['losses/explained_variance']:5.2f}"
        )

        if iteration % args.eval_interval == 0:
            print(f"iter {iteration}/{args.num_iterations}  eval_return={run_eval():.3f}")

    # Only if the loop's last iteration was not itself an eval point.
    if args.num_iterations % args.eval_interval:
        print(f"final eval_return={run_eval():.3f}")

    torch.save(agent.state_dict(), run_dir / "agent.pt")

    eval_file.close()
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
