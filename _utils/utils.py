import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
import os

def get_last_experiment_name(algo, env):
    count = sub_dir_count(f"{algo}/logs/{env}/") - 1
    return f"experiment_{count}"


def sub_dir_count(path):
    count = 0
    try:
        for f in os.listdir(path):
            child = os.path.join(path, f)
            if os.path.isdir(child) and child != "__pycache__":
                print(child)
                count += 1
    except FileNotFoundError:
        pass

    return count


def compute_returns_for_one_episode(rewards: torch.tensor, gamma: float):
    if rewards.nelement() == 0:
        return torch.empty_like(rewards)
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


class OUNoise(object):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
