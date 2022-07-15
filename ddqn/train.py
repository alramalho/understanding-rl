import time

from utils import plot_rwrds_and_losses
import gym
from agent import DoubleDQNAgent
import numpy as np
import torch


def train(config):
    start = time.time()
    env = gym.make(config["problem"])
    if config["random_seed"]:
        torch.manual_seed(config["random_seed"])
        np.random.seed(config["random_seed"])

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = DoubleDQNAgent(env, input_dim, output_dim, config)
    ep_rewards, losses = [], []

    for episode in range(config["n_episodes"]):
        reward, loss = agent.run_episode()
        ep_rewards.append(reward)
        losses.append(loss)

        if episode % config["print_freq"] == 0:
            r = round(float(np.mean(ep_rewards[-config["print_freq"]:])), 2)
            l = round(float(np.mean(losses[-config["print_freq"]:])), 2)
            print("Episode {}, Average Reward {}, Average Loss: {}".format(episode, r, l))

    end = time.time()
    extra_config = {
        "agent": type(agent).__name__,
        "execution time": start-end
    }
    plot_rwrds_and_losses(
        rewards=ep_rewards,
        losses=losses,
        config=dict(config, **extra_config),
        roll=30
    )
