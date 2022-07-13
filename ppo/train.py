import gym
import torch

from agent import PPOAgent
from utils import plot_rwrds_and_losses
import numpy as np


def train(config):
    env = gym.make(config["problem"])
    if config["random_seed"]:
        torch.manual_seed(config["random_seed"])
        env.seed(config["random_seed"])
        np.random.seed(config["random_seed"])

    if config["has_continuous_actions"]:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    if config["has_continuous_space"]:
        space_dim = env.observation_space.shape[0]
    else:
        space_dim = env.observation_space.n

    agent = PPOAgent(env, space_dim, action_dim, config)
    entropy_reg = config["entropy_reg"]

    ep_rewards, losses = [], []

    for episode in range(1, config["n_episodes"] + 1):
        reward = agent.run_episode()

        ep_rewards.append(reward)

        if episode % config["ppo_update_freq"] == 0:
            loss = agent.update()
            losses.append(loss)

        if episode % config["entropy_decay_freq"] == 0:
            entropy_reg = 0.9 * entropy_reg

        if episode % config["action_std_decay_freq"] == 0:
            agent.decay_action_std()

        if episode % config["print_freq"] == 0:
            r = round(float(np.mean(ep_rewards[-config["print_freq"]:])), 2)
            loss_freq = int(config["print_freq"] / config["ppo_update_freq"])
            l = round(float(np.mean(losses[-loss_freq:])), 2)
            print("Episode {}, Average Reward {}, Average Loss: {}".format(episode, r, l))

    plot_rwrds_and_losses(ep_rewards, losses, config)


if __name__ == "__main__":
    train()
