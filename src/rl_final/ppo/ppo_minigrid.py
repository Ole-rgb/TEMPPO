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

    # Per-seed subdir so a --multirun sweep drops eval_rewards.csv / agent.pt where the
    # analysis code expects them: <base>/seed_<N>/
    run_dir = Path(HydraConfig.get().runtime.output_dir) / f"seed_{args.seed}"
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

    eval_file = open(run_dir / "eval_rewards.csv", "w", newline="")
    eval_csv = csv.writer(eval_file)
    eval_csv.writerow(["eval_steps", "eval_rewards"])

    def run_eval():
        # One mp4 per eval, in a step-named folder, so you can watch the policy
        # improve across training instead of overwriting a single file.
        video_dir = run_dir / "videos" / f"step_{global_step:08d}" if args.capture_video else None
        mean_return = evaluate(agent, args.env_id, args.eval_episodes, device, video_dir)
        eval_csv.writerow([global_step, mean_return])
        eval_file.flush()
        writer.add_scalar("charts/eval_return", mean_return, global_step)
        return mean_return

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(
        device
    )
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(
        device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    terminateds = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_terminated = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            terminateds[step] = next_terminated

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done, next_terminated = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
                torch.Tensor(terminations).to(device),
            )
            if "episode" in infos:
                mask = infos["_episode"]
                rs, ls = infos["episode"]["r"][mask], infos["episode"]["l"][mask]
                for r, ln in zip(rs, ls, strict=False):
                    writer.add_scalar("charts/episodic_return", r, global_step)
                    writer.add_scalar("charts/episodic_length", ln, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonboundary = 1.0 - next_done
                    nextnonterminal = 1.0 - next_terminated
                    nextvalues = next_value
                else:
                    nextnonboundary = 1.0 - dones[t + 1]
                    nextnonterminal = 1.0 - terminateds[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonboundary * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for _epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        print(
            f"iter {iteration:4d}/{args.num_iterations}  "
            f"steps {global_step:>8,}  "
            f"ent {entropy_loss.item():5.3f}  kl {approx_kl.item():.4f}  "
            f"clip {np.mean(clipfracs):.3f}  ev {explained_var:5.2f}"
        )

        if iteration % args.eval_interval == 0:
            print(f"iter {iteration}/{args.num_iterations}  eval_return={run_eval():.3f}")

    print(f"final eval_return={run_eval():.3f}")
    torch.save(agent.state_dict(), run_dir / "agent.pt")

    eval_file.close()
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
