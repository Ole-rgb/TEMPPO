import hashlib
from os.path import join

import gymnasium as gym
import hydra
import torch
from gymnasium.wrappers import RecordEpisodeStatistics
from minigrid.wrappers import ImgObsWrapper
from omegaconf import DictConfig
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def make_key(mission: str, environment: str) -> str:
    return hashlib.sha256(make_prompt(mission, environment).encode()).hexdigest()

def make_prompt(mission: str, environment: str) -> str:
    return f"Mission: {mission}\nMap:\n{environment}"

OBJ   = {"wall":"W","floor":"F","door":"D","key":"K","ball":"A","box":"B","goal":"G","lava":"V"}
ADIR  = {0:">",1:"V",2:"<",3:"^"}
CCHAR = {"red":"R","green":"G","blue":"B","purple":"P","yellow":"Y","grey":"E"}

def render_map(base):
    g = base.grid
    rows = []
    for j in range(g.height):
        row = []
        for i in range(g.width):
            if (i, j) == tuple(base.agent_pos):
                row.append(ADIR[base.agent_dir] * 2)
            else:
                o = g.get(i, j)
                if o is None:
                    row.append("  ")
                elif o.type == "door":
                    row.append("__" if o.is_open
                               else (("L" if o.is_locked else "D") + CCHAR[o.color]))
                else:
                    row.append(OBJ[o.type] + CCHAR[o.color])
        rows.append("".join(row))
    return "\n".join(rows)

class Agent(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.policy = nn.Linear(1024, 7)  # actor
        self.value = nn.Linear(1024, 1)  # critic: one scalar V(s)

    def forward(self, x):
        h = self.hidden(x)
        return self.policy(h), self.value(h)  # logits, value


def to_t(o):
    """
    Convert the env observation to a tensor suitable for the neural network.
    """
    return torch.as_tensor(o, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


def discounted_rewards(rewards, gamma, bootstrap=0):
    """
    Calculate discounted rewards and return them as a tensor
    """
    returns, G = [], bootstrap
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)

    env = RecordEpisodeStatistics(ImgObsWrapper(gym.make(cfg.env.gym_id)))

    writer = SummaryWriter(
        join(hydra.utils.get_original_cwd(),
        f"runs/{cfg.env.gym_id}__seed{cfg.seed}")
    )

    agent = Agent()

    obs, _ = env.reset(seed=cfg.seed)
    optim = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate)
    log_probs, entropies, values, rewards = [], [], [], []

    ################################################
    seen = set()
    seen.add(make_key(env.unwrapped.mission, render_map(env.unwrapped)))  # first episode too
    ################################################

    for global_step in tqdm(range(cfg.total_env_steps // 2)):

        logits, value = agent(to_t(obs))
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()

        obs, reward, terminated, truncated, info = env.step(action.item())

        log_probs.append(action_dist.log_prob(action))
        entropies.append(action_dist.entropy())
        values.append(value)
        rewards.append(reward)

        if terminated or truncated:
            if truncated and not terminated:  # not a real terminal
                with torch.no_grad():
                    _, next_value = agent(to_t(obs))
                bootstrap = next_value.item()
            else:
                bootstrap = 0.0

            returns = discounted_rewards(rewards, cfg.gamma, bootstrap)
            values = torch.cat(values).squeeze(-1)

            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            log_probs = torch.cat(log_probs)
            entropies = torch.cat(entropies)

            policy_loss = -(log_probs * advantages).mean()
            value_loss = nn.MSELoss()(values, returns)

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropies.mean()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            optim.step()

            # Logging
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropies.mean().item(), global_step)

            # reset
            log_probs, values, entropies, rewards = [], [], [], []
            obs, _ = env.reset()

            ################################################
            seen.add(make_key(env.unwrapped.mission, render_map(env.unwrapped)))
            ################################################

    env.close()
    writer.close()

    ################################################
    print(len(seen), "unique layouts")
    ################################################

    # # Watch the trained agent
    # env = ImgObsWrapper(gym.make(cfg.env.gym_id, render_mode="human"))
    # obs, _ = env.reset(seed=cfg.seed)
    # for _ in range(3):
    #     done = False
    #     while not done:
    #         with torch.no_grad():
    #             logits, value = agent(to_t(obs))
    #             # action = logits.argmax(dim=-1) # greedy
    #             action_dist = torch.distributions.Categorical(logits=logits)
    #             action = action_dist.sample()
    #         obs, _, terminated, truncated, _ = env.step(action.item())
    #         done = terminated or truncated
    #     obs, _ = env.reset()

    # env.close()


if __name__ == "__main__":
    main()
