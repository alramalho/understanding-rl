import random
from collections import deque
from utils import plot_rwrds_and_losses

import gym
import numpy as np
import torch
from torch import nn

config = {
    "n_episodes": 2000,
    "print_freq": 10,
    "max_steps": 200,
    "batch_size": 32,
    "buffer_max_capacity": 100000,
    "init_epsilon": 0.8,
    "min_epsilon": 0.01,
    "epsilon_decay_freq": 50,  # in steps
    "gamma": 0.99,
    "learning_rate": 0.00025,
    "tau": 1000,  # in steps
}
epsilon = config["init_epsilon"]


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

        # actions and states need to have same rows
        return torch.tensor(states).float(), torch.tensor(actions).view(-1, 1), torch.tensor(
            rewards).float(), torch.tensor(
            next_states).float(), torch.tensor(dones).float()

    def store(self, tuple):
        self.mem.append(tuple)

    def __len__(self):
        return len(self.mem)


class Brain(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Brain, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        self.q_target = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=config["learning_rate"])

    def target_qvalue(self, states, single=False):
        if single:
            states = states.unsqueeze(0)
        return self.q_target(states)

    def qvalue(self, states, single=False):
        if single:
            states = states.unsqueeze(0)

        return self.q_net(states)

    def transfer_weights(self):
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DoubleDQNAgent:
    def __init__(self, env, state_dim, action_dim):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.overall_step = 0
        self.brain = Brain(state_dim, action_dim)
        self.buffer = ReplayBuffer(config["buffer_max_capacity"])

        self.loss_criterion = nn.HuberLoss()

    def e_greedy_policy(self, state):
        if random.random() < epsilon:
            action = random.choice(range(self.action_dim))
        else:
            action = torch.argmax(self.brain.qvalue(state, single=True).detach())
            action = action.item()

        return action

    def train(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(config["batch_size"])

        y = torch.zeros((len(states), 1))

        for i in range(len(states)):
            if dones[i]:
                y[i] = rewards[i]
            else:
                y[i] = rewards[i] + config["gamma"] * torch.max(self.brain.target_qvalue(next_states[i], True).detach())

        state_action_values = self.brain.qvalue(states).gather(1, actions)
        assert state_action_values.shape == y.shape
        loss = self.loss_criterion(state_action_values, y)

        self.brain.update(loss)

        return loss.detach().numpy()

    def run_episode(self):
        global epsilon
        losses, acc_reward = [], 0

        s = self.env.reset()
        for step in range(config["max_steps"]):
            self.overall_step += 1
            a = self.e_greedy_policy(torch.tensor(s).float())

            s_, r, done, _ = self.env.step(a)
            acc_reward += r

            self.buffer.store(tuple=(s, a, r, s_, done))

            if len(self.buffer) >= config["batch_size"]:
                loss = self.train()
                losses.append(loss)

            if self.overall_step % config["tau"] == 0:
                self.brain.transfer_weights()

            if step % config["epsilon_decay_freq"] == 0:
                epsilon = max(config["min_epsilon"], 0.95 * epsilon)

            s = s_

            if done:
                break

        return acc_reward, np.mean(losses)


def main():
    ep_rewards, losses = [], []

    env = gym.make("CartPole-v1")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = DoubleDQNAgent(env, input_dim, output_dim)

    for episode in range(config["n_episodes"]):
        reward, loss = agent.run_episode()
        ep_rewards.append(reward)
        losses.append(loss)

        if episode % config["print_freq"] == 0:
            r = round(float(np.mean(ep_rewards[-config["print_freq"]:])), 2)
            l = round(float(np.mean(losses[-config["print_freq"]:])), 2)
            e = round(epsilon, 2)
            print("Episode {}, Average Reward {}, Average Loss: {}, Epsilon {}".format(episode, r, l, e))

    plot_rwrds_and_losses(rewards=ep_rewards, losses=losses, config=config, roll=30)


if __name__ == "__main__":
    main()
