import matplotlib.pyplot as plt
import sys
import numpy as np
import torch


def compute_returns_for_one_episode(rewards: torch.tensor, gamma: float):
    if rewards.nelement() == 0: return torch.empty_like(rewards)
    result = [rewards[-1]]
    for reward in reversed(rewards[:-1]):
        new = reward + gamma * result[0]
        result.insert(0, new)
    return torch.tensor(result).view(len(result), 1)

def compute_returns_for_several_episodes(rewards, dones, gamma):
    result = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        result.insert(0, discounted_reward)
    return torch.tensor(result, dtype=torch.float32)


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


def roll_avg(array, window):
    avgs = []
    for i in range(len(array)):
        if i < window:
            avgs.append(np.mean(array[0: i]))
        else:
            avgs.append(np.mean(array[i - window: i]))

    return avgs


def listify(config):
    extra_message = sys.argv[sys.argv.index('-m') + 1] if '-m' in sys.argv else ''
    title_str = '\n'.join([str(k) + ': ' + str(v) for k, v in config.items()])
    return f'{title_str}\n{extra_message}'


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class SimpleBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]

