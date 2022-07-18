import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from _utils.utils import Bcolors


def listify(config):
    extra_message = sys.argv[sys.argv.index('-m') + 1] if '-m' in sys.argv else ''
    title_str = '\n'.join([str(k) + ': ' + str(v) for k, v in config.items()])
    return f'{title_str}\n{extra_message}'


def roll_avg(array, window):
    avgs = []
    for i in range(len(array)):
        if i < window:
            avgs.append(np.mean(array[0: i]))
        else:
            avgs.append(np.mean(array[i - window: i]))

    return avgs


def plot_rwrds_and_losses(rewards, losses=None, config=None, roll=5):
    fig, ax = plt.subplots(ncols=2, constrained_layout=True, figsize=(10, 5))
    ax[0].plot(range(len(rewards)), rewards, label='Rewards', c='g', alpha=0.5)
    ax[0].plot(range(len(rewards)), roll_avg(rewards, roll), label=f'{roll} Ep Avg Reward', c='lime', alpha=0.7)
    ax[0].set_xlabel('Episode')
    ax[0].legend(loc='lower right')

    if losses is not None:
        ax[1].plot(range(len(losses)), losses, label='Loss', c='r')
        ax[1].legend(loc='upper right')
        ax[1].set_xlabel('Episode')

    if config is not None:
        fig.suptitle(listify(config))

    plt.show()


def plot_rwrds_and_aclosses(rewards, a_losses=None, c_losses=None, ac_losses=None, config=None, roll=5):
    fig, ax = plt.subplots(ncols=2, constrained_layout=True, figsize=(10, 5))
    ax[0].plot(range(len(rewards)), rewards, label='Rewards', c='g', alpha=0.5)
    ax[0].plot(range(len(rewards)), roll_avg(rewards, roll), label=f'{roll} Ep Avg Reward', c='lime', alpha=0.7)
    ax[0].set_xlabel('Episode')
    ax[0].legend(loc='lower right')

    if a_losses is not None:
        ax[1].plot(range(len(a_losses)), a_losses, label='Actor Loss', c='r', alpha=0.7)
        ax[1].legend(loc='upper right')
    if c_losses is not None:
        ax[1].plot(range(len(c_losses)), c_losses, label='Critic Loss', c='orange', alpha=0.7)
        ax[1].legend(loc='upper right')
    if ac_losses is not None:
        ax[1].plot(range(len(ac_losses)), ac_losses, label='Actor Critic Loss', c='blue', alpha=0.7)
        ax[1].legend(loc='upper right')
        ax[1].set_xlabel('Episode')

    if config is not None:
        fig.suptitle(listify(config))

    plt.show()

def plot_df(results_df: pd.DataFrame, config):
    print(f'{Bcolors.OKGREEN}Plotting{Bcolors.ENDC}')
    plot_rwrds_and_losses(
        rewards=results_df["reward"].values.tolist(),
        losses=results_df["loss"].values.tolist(),
        config=config,
        roll=30
    )


def plot(algo: str, env: str, exp: str):
    path = f"{algo}/logs/{env}/{exp}"
    result_df = pd.read_csv(f"{path}/results.csv")
    plot_df(results_df=result_df, config={"file": path})

