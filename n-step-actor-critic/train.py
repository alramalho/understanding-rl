import gym
import numpy as np
from utils import plot_rwrds_and_aclosses
from agent import NStepActorCriticAgent
import torch


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

    agent = NStepActorCriticAgent(env, space_dim, action_dim, config)

    rewards, a_losses, c_losses = [], [], []

    for episode in range(config["num_episodes"]):
        reward, a_loss, c_loss = agent.run_episode()

        rewards.append(reward)
        a_losses.append(a_loss)
        c_losses.append(c_loss)

        if episode % config["print_freq"] == 0:
            r = np.mean(rewards[-config["print_freq"]:])
            al = round(float(np.mean(a_losses[-config["print_freq"]:])), 2)
            cl = round(float(np.mean(c_losses[-config["print_freq"]:])), 2)
            print("Episode {} \t Avg Reward {} \t Actor Loss {} \t Critic Loss {}".format(episode, r, al, cl))

    plot_rwrds_and_aclosses(rewards, a_losses, c_losses, roll=50, config=dict(config, **{"agent": type(agent).__name__}))
