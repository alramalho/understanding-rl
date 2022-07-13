import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import compute_returns_for_one_episode


class Brain(nn.Module):

    def __init__(self, n_states, n_actions):
        super(Brain, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.c1 = nn.Linear(n_states, 256)
        self.c2 = nn.Linear(256, 1)
        self.a1 = nn.Linear(n_states, 256)
        self.a2 = nn.Linear(256, n_actions)

    def value(self, states):
        v = nn.Sequential(self.c1, nn.ReLU(), self.c2)
        return v(states)

    def action_probs(self, states):
        v = nn.Sequential(self.a1, nn.ReLU(), self.a2)
        return F.softmax(v(states))


class ReinforceWithBaselineAgent:

    def __init__(self, env, input_dim, output_dim, config):
        self.env = env
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.brain = Brain(input_dim, output_dim)
        self.config = config

        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=self.config["alpha"])

    def update(self, states, actions, rewards, entropy_term):
        # torch data treatment
        actions = actions.view(len(actions), 1)

        returns = compute_returns_for_one_episode(rewards, self.config["gamma"])

        advantage = returns - self.brain.value(states)  # must be shape n_steps x 1
        a_log_probs = torch.log(self.brain.action_probs(states)).gather(1, actions)  # must be shape n_steps x 1

        a_loss = (-a_log_probs * advantage).mean()
        c_loss = (0.5 * advantage.pow(2)).mean()  # so that the derivative of this is just the advantage fn

        ac_loss = a_loss + c_loss + self.config["entropy_reg"] * entropy_term

        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()

        return ac_loss

    def run_episode(self):

        s = self.env.reset()

        states = []
        actions = []
        rewards = []

        ac_losses = []
        entropy_term = 0
        for step in range(self.config["max_steps"]):
            a_dist = self.brain.action_probs(torch.tensor(s).float().reshape(1, -1)).detach().numpy()

            a = np.random.choice(self.output_dim, p=np.squeeze(a_dist))

            entropy_term += -np.sum(np.mean(a_dist) * np.log(a_dist))

            s_, r, done, _ = self.env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)

            s = s_

            if done:
                t = torch.tensor
                states = t(states).float().view(-1, self.input_dim)
                ac_l = self.update(states, t(actions), t(rewards), entropy_term)
                ac_losses.append(ac_l.detach().numpy())
                break

        return np.sum(rewards), np.mean(ac_losses)
