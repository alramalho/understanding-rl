import optuna
import pandas as pd
from optuna.visualization import plot_intermediate_values
import gym
from agent import DoubleDQNAgent
import numpy as np
import torch
from datetime import datetime
from utils import plot_rwrds_and_losses, Bcolors
from atari_wrappers import AtariWrapper
from trainlogger import TrainLogger


def create(execution_config, agent_config):
    print(f'{Bcolors.OKGREEN}Creating{Bcolors.ENDC}')

    env = gym.make(execution_config["problem"])
    if execution_config["problem"] == "PongNoFrameskip-v4":
        env = AtariWrapper(env)

    if execution_config["random_seed"]:
        torch.manual_seed(execution_config["random_seed"])
        np.random.seed(execution_config["random_seed"])

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = DoubleDQNAgent(env, input_dim, output_dim, agent_config)

    return agent


def train(execution_config, agent, experiment_title, trial_number=None, is_trial=False):
    # Create Train Logger
    output = experiment_title
    if is_trial:
        output = output + f"/trial_{trial_number}"

    logger = TrainLogger(output=output)
    logger.log_config(agent, dict(execution_config, **agent.config))

    # Train
    print(f'{Bcolors.OKGREEN}Training{Bcolors.ENDC}')
    try:
        ep_rewards, losses, epsilons = [], [], []
        for episode in range(execution_config["n_episodes"]):
            reward, loss, epsilon = agent.run_episode()

            logger.log_to_results(",".join([str(round(reward, 2)), str(round(loss, 2)), str(round(epsilon, 2))]) + '\n')

            ep_rewards.append(reward)
            losses.append(loss)
            epsilons.append(epsilon)

            if episode % execution_config["print_freq"] == 0:
                r = round(float(np.mean(ep_rewards[-execution_config["print_freq"]:])), 2)
                l = round(float(np.mean(losses[-execution_config["print_freq"]:])), 2)
                e = round(float(np.mean(epsilons[-execution_config["print_freq"]:])), 2)
                print("Episode {}, Average Reward {}, Average Loss: {}, Average Epsilon {}".format(episode, r, l, e))

    except KeyboardInterrupt:
        pass

    return logger.get_results_df()


def plot(results_df: pd.DataFrame, config):
    print(f'{Bcolors.OKGREEN}Plotting{Bcolors.ENDC}')
    plot_rwrds_and_losses(
        rewards=results_df["reward"].values.tolist(),
        losses=results_df["loss"].values.tolist(),
        config=config,
        roll=30
    )
