import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
from utils import compute_returns_for_several_episodes, SimpleBuffer
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
            {'params': self.actor.parameters(), 'lr': self.config["lr_actor"]},
            {'params': self.critic.parameters(), 'lr': self.config["lr_critic"]}
        ])

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
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
            self.action_std = round(min(self.config["min_action_std"], self.action_std * self.config["action_std_decay_freq"]), 4)
        else:
            raise ValueError("This function shouldn't be called")


class PPOAgent:

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

    def update(self):
        states = torch.tensor(self.buffer.states).float()
        rewards = torch.tensor(self.buffer.rewards).float()
        actions = torch.tensor(self.buffer.actions).float()
        if self.config["has_continuous_actions"]:
            actions = actions.view(-1, self.action_dim)
        old_log_probs = torch.tensor(self.buffer.logprobs).float()
        dones = torch.tensor(self.buffer.dones).float()
        returns = compute_returns_for_several_episodes(rewards, dones, self.config["gamma"])
        returns = (returns - returns.mean()) / returns.std() + 1e-7

        losses = []
        for _ in range(self.config["K_epochs"]):
            dist, dist_entropy = self.brain.get_action_dist(states)
            log_probs = dist.log_prob(actions)
            ratio = log_probs.exp() / old_log_probs.exp()

            state_values = self.brain.critic(states).squeeze()
            advantage = returns - state_values.detach()
            # advantage = (advantage - advantage.mean()) / advantage.std() + 1e-5

            a_loss = -torch.min(ratio * advantage, torch.clamp(ratio, 1 - self.config["epsilon"], 1 + self.config["epsilon"]) * advantage)

            c_loss = self.mse_loss(state_values, returns)

            loss = (a_loss + 0.5 * c_loss + self.config["entropy_reg"] * dist_entropy).mean()

            self.brain.update(loss)
            losses.append(loss.detach().numpy())

        self.buffer.clear()
        return np.mean(losses)

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
            self.buffer.dones.append(done)
            rewards.append(float(r))
            self.buffer.actions.append(a)
            self.buffer.logprobs.append(a_log_prob)

            s = s_

            if done:
                break

        return np.sum(rewards)
