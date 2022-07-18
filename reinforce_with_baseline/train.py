import gym
from agent import ReinforceWithBaselineAgent
from utils.utils import plot_rwrds_and_losses
import numpy as np


def train(config):
    env = gym.make("CartPole-v1")  # gym.make("FrozenLake-v1", map_name='4x4', is_slippery=False)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = ReinforceWithBaselineAgent(env, input_dim, output_dim, config)

    rewards = []
    losses = []

    for episode in range(config["num_episodes"]):
        reward, ac_loss = agent.run_episode()

        rewards.append(reward)
        losses.append(ac_loss)

        if episode % config["print_freq"] == 0:
            print("Episode {} \t\t Avg Reward {} \t\t Avg Loss {}".format(
                episode,
                np.mean(rewards[-config["print_freq"]:]),
                np.mean(losses[-config["print_freq"]:]))
            )

    plot_rwrds_and_losses(rewards, losses=losses, config=config, roll=100)
