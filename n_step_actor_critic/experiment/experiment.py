import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
from utils.utils import SimpleBuffer
import numpy as np


class Brain(nn.Module):

    def __init__(self, state_dim, action_dim, config):
        super(Brain, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        if self.config["has_continuous_actions"]:
            self.action_std = self.config["action_std"]

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.config["learning_rate"]},
            {'params': self.critic.parameters(), 'lr': self.config["learning_rate"]}
        ])

    def update(self, c_loss, a_loss):
        self.optimizer.zero_grad()
        a_loss.backward()
        c_loss.backward()
        self.optimizer.step()

    def get_action_dist(self, states, single=True):
        if self.config["has_continuous_actions"]:
            a_mean = self.actor(states)
            a_variance = torch.full((self.action_dim,), self.action_std ** 2)
            a_variance = a_variance.expand_as(a_mean)

            if single:
                cov_mat = torch.diag(a_variance).unsqueeze(0)
            else:
                cov_mat = torch.diag_embed(a_variance)

            dist = MultivariateNormal(a_mean, cov_mat)
        else:
            a_probs = F.softmax(self.actor(states))
            dist = Categorical(a_probs)

        return dist, dist.entropy()

    def decay_action_std(self):
        if self.config["has_continuous_actions"]:
            self.action_std = round(
                min(self.config["min_action_std"], self.action_std * self.config["action_std_decay_freq"]), 4)
        else:
            raise ValueError("This function shouldn't be called")


class NStepActorCriticAgent:

    def __init__(self, env, state_dim, action_dim, config):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = SimpleBuffer()
        self.brain = Brain(state_dim, action_dim, config)
        self.config = config

        self.mse_loss = nn.MSELoss()

    def decay_action_std(self):
        self.brain.decay_action_std()

    def get_action_probs(self, states, actions):
        dist, _ = self.brain.get_action_dist(states)
        return dist.log_prob(actions)

    def get_action(self, state):
        dist, _ = self.brain.get_action_dist(state.unsqueeze(0), single=True)

        a = dist.sample()
        a_log_prob = dist.log_prob(a)

        return a.detach(), a_log_prob.detach()

    # Barto p.143
    def compute_n_step_returns(self, states, rewards, gamma):
        R = []
        T = len(rewards)
        n_steps = self.config["n_steps"]
        for t in range(T):
            R_t = 0
            for n in range(n_steps + 1):
                if t + n >= T:
                    continue
                else:
                    R_t += gamma ** n * rewards[t + n]
                    if n == n_steps:
                        R_t += gamma ** n * self.brain.critic(states[t + n].unsqueeze(0)).data.detach().item()
            R.append(R_t)

        R = torch.tensor(R)
        assert R.shape == rewards.shape
        return R

    def update(self):
        states = torch.tensor(self.buffer.states).float()
        rewards = torch.tensor(self.buffer.rewards).float()
        actions = torch.tensor(self.buffer.actions).float()
        if self.config["has_continuous_actions"]:
            actions = actions.view(-1, self.action_dim)

        returns = self.compute_n_step_returns(states, rewards, self.config["gamma"])
        dist, dist_entropy = self.brain.get_action_dist(states)
        log_probs = dist.log_prob(actions)
        state_values = self.brain.critic(states).squeeze()
        advantage = returns - state_values

        a_loss = (-log_probs * advantage.detach()).mean()
        c_loss = advantage.pow(2).mean()

        self.brain.update(a_loss, c_loss)

        self.buffer.clear()
        return a_loss.detach().numpy(), c_loss.detach().numpy()

    def run_episode(self):
        rewards = []
        if self.config["random_seed"]:
            s = self.env.reset(seed=self.config["random_seed"])
        else:
            s = self.env.reset()

        while True:
            a, a_log_prob = self.get_action(torch.FloatTensor(s))
            if not self.config["has_continuous_actions"]:
                a = a.item()
            else:
                a = a.numpy().reshape((self.action_dim,))

            s_, r, done, info = self.env.step(a)

            self.buffer.states.append(s)
            self.buffer.rewards.append(r)
            self.buffer.actions.append(a)
            rewards.append(r)

            s = s_

            if done:
                break

        a_loss, c_loss = self.update()

        return np.sum(rewards), a_loss, c_loss
