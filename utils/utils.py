import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
import os


def sub_dir_count(path):
    count = 0
    for f in os.listdir(path):
        child = os.path.join(path, f)
        if os.path.isdir(child) and child != "__pycache__":
            print(child)
            count += 1
    return count


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
