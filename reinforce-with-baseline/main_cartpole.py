import matplotlib.pyplot as plt
from utils import plot_rwrds_and_losses
import gym
import numpy as np
from agent import ReinforceWithBaselineAgent

main_config = {
    "num_episodes": 1000,
    "print_freq": 10  # in episodes
}

agent_config = {
    "max_steps": 501,
    "alpha": 4e-4,
    "beta": 4e-4,
    "gamma": 0.99,
    "entropy_reg": 0.1
}

config = dict(main_config, **agent_config)


def main():
    env = gym.make("CartPole-v1")  # gym.make("FrozenLake-v1", map_name='4x4', is_slippery=False)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = ReinforceWithBaselineAgent(env, input_dim, output_dim, config)

    rewards = []
    losses = []

    for episode in range(main_config["num_episodes"]):
        reward, ac_loss = agent.run_episode()

        rewards.append(reward)
        losses.append(ac_loss)

        if episode % main_config["print_freq"] == 0:
            print("Episode {} \t\t Avg Reward {} \t\t Avg Loss {}".format(
                episode,
                np.mean(rewards[-main_config["print_freq"]:]),
                np.mean(losses[-main_config["print_freq"]:]))
            )

    plot_rwrds_and_losses(rewards, losses=losses, config=config, roll=100)



if __name__ == '__main__':
    main()
