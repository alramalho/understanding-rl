import gym
import numpy as np
import torch
import torch.nn as nn
import random


class Brain:

    def __init__(self, obs_dim, act_dim, config):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config

        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, act_dim)
        )

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.config["learning_rate"])

    def qvalue(self, state):
        # Pytorch idiosyncrasy – onlxy supports batches as input
        state = state.unsqueeze(0)

        return self.q_net(state).squeeze()

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class SemiGradientSarsaAgent:

    def __init__(self, env, obs_dim, act_dim, config):
        self.env = env
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config
        self.epsilon = config["epsilon"]

        self.brain = Brain(obs_dim, act_dim, config)

    def select_action(self, state):
        # Epsilon - greedy
        if random.random() <= self.epsilon:
            action = random.choice(range(self.act_dim))
        else:
            action = torch.argmax(self.brain.qvalue(state))
        return int(action)

    def decay_epsilon(self):
        min_eps = 0.05
        reduction = (self.epsilon - min_eps) / (self.config["n_episodes"] * 0.7)
        self.epsilon = self.epsilon - reduction

    def run_episode(self):
        s = torch.FloatTensor(self.env.reset())
        a = self.select_action(s)

        ep_reward = 0
        losses = []
        while True:

            s_, r, done, info = self.env.step(a)
            s_ = torch.FloatTensor(s_)
            ep_reward += r

            a_ = self.select_action(s_)

            if done:
                loss = 0.5*(r - self.brain.qvalue(s)[a])**2
                self.brain.update(loss)
                losses.append(loss.detach().numpy())
                break

            loss = 0.5 * (r + self.config["gamma"] * self.brain.qvalue(s_)[a_] - self.brain.qvalue(s)[a]) ** 2
            self.brain.update(loss)
            losses.append(loss.detach().numpy())

            s = s_
            a = a_
            self.decay_epsilon()

        return ep_reward, np.mean(losses)
