import random
from collections import deque
import numpy as np
import torch
from torch import nn


class ReplayBuffer:

    def __init__(self, max_capacity):
        self.mem = deque([], maxlen=max_capacity)

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for s, a, r, s_, done in random.sample(self.mem, min(batch_size, len(self.mem))):
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_)
            dones.append(done)

        return torch.tensor(states).float(), torch.tensor(actions).view(-1, 1), torch.tensor(
            rewards).float(), torch.tensor(
            next_states).float(), torch.tensor(dones).float()

    def store(self, tuple):
        self.mem.append(tuple)

    def __len__(self):
        return len(self.mem)


class DQNConv(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=state_dim, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=32 * 9 * 9, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_dim)
        )

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)  # convert to 2D
        return self.fc(conv_out)


class Brain(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Brain, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        if config["net_arch"] in ["mlp_small", "mlp_medium"]:
            hidden_layer_size = 64 if config["net_arch"] == "mlp_small" else 256

            self.q_net = nn.Sequential(
                nn.Linear(state_dim, hidden_layer_size), nn.ReLU(),
                nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(),
                nn.Linear(hidden_layer_size, action_dim)
            )

            self.q_target = nn.Sequential(
                nn.Linear(state_dim, hidden_layer_size), nn.ReLU(),
                nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(),
                nn.Linear(hidden_layer_size, action_dim)
            )
        elif config["net_arch"] == "conv":
            self.q_net = DQNConv(state_dim, action_dim)
            self.q_target = DQNConv(state_dim, action_dim)
        else:
            raise ValueError("Invalid net_arch")

        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=self.config["learning_rate"])

    def target_qvalue(self, states):
        if len(states.shape) == 1:  # Pytorch doesn't accept single inputs
            states = states.unsqueeze(0)
        return self.q_target(states)

    def qvalue(self, states):
        if len(states.shape) == 1:  # Pytorch doesn't accept single inputs
            states = states.unsqueeze(0)
        return self.q_net(states)

    def transfer_weights(self):
        params = self.q_net.parameters()
        target_params = self.q_target.parameters()
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - self.config["tau"])
            torch.add(target_param.data, param.data, alpha=self.config["tau"], out=target_param.data)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DDQNAgent:
    def __init__(self, env, state_dim, action_dim, config):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        self.training_step = 0
        self.brain = Brain(state_dim, action_dim, config)
        self.buffer = ReplayBuffer(self.config["buffer_max_capacity"])
        self.epsilon = self.config["initial_epsilon"]
        self.loss_criterion = nn.HuberLoss()

    def decay_epsilon(self):
        initial = self.config["initial_epsilon"]
        final = self.config["final_epsilon"]
        fraction = self.config["exploration_fraction"]
        n_episodes = self.config["n_episodes"]

        self.epsilon = max(self.epsilon - np.abs(final - initial) / (fraction * n_episodes), final)

    def e_greedy_policy(self, state):
        if random.random() < self.epsilon:
            action = random.choice(range(self.action_dim))
        else:
            action = torch.argmax(self.brain.qvalue(state).detach())
            action = action.item()

        return action

    def train(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.config["batch_size"])

        y = torch.zeros((len(states), 1))

        for i in range(len(states)):
            if dones[i]:
                y[i] = rewards[i]
            else:
                y[i] = rewards[i] + self.config["gamma"] * torch.max(self.brain.target_qvalue(next_states[i]).detach())

        state_action_values = self.brain.qvalue(states).gather(1, actions)
        assert state_action_values.shape == y.shape
        loss = self.loss_criterion(state_action_values, y)

        self.brain.update(loss)

        return loss.detach().numpy()

    def run_episode(self):
        losses, acc_reward = [], 0

        s = self.env.reset()
        while True:
            a = self.e_greedy_policy(torch.tensor(s).float())

            s_, r, done, _ = self.env.step(a)
            acc_reward += r

            self.buffer.store(tuple=(s, a, r, s_, done))

            if len(self.buffer) >= self.config["batch_size"]:
                loss = self.train()
                losses.append(loss)

            if self.training_step % self.config["target_update_interval"] == 0:
                self.brain.transfer_weights()

            s = s_
            self.training_step += 1

            if done:
                break

        self.decay_epsilon()

        return acc_reward, np.mean(losses)
