from collections import deque
import random
import torch
import torch.nn as nn
import numpy as np
from _utils.utils import OUNoise


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def store(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        ft = torch.FloatTensor
        return ft(state_batch), ft(action_batch), ft(reward_batch), ft(next_state_batch), ft(done_batch)

    def __len__(self):
        return len(self.buffer)


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, config):
        super(Critic, self).__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(n_states + n_actions, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def qvalue(self, state, action):
        return self.net(torch.cat([state, action], -1))


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, config):
        super(Actor, self).__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(n_states, 256), nn.ReLU(),
            nn.Linear(256, n_actions), nn.Tanh()
        )

    def policy(self, state):
        return self.net(state)


class Brain:

    def __init__(self, input_dim, output_dim, config):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config

        self.actor = Actor(input_dim, output_dim, config)
        self.t_actor = Actor(input_dim, output_dim, config)
        self.critic = Critic(input_dim, output_dim, config)
        self.t_critic = Critic(input_dim, output_dim, config)

        self.update_targets(tau=1)

        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=config["actor_lr"])
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=config["critic_lr"])

    def update_targets(self, tau=0.1):
        for target_param, param in zip(self.t_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.t_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self, a_loss, c_loss):
        self.a_opt.zero_grad()
        a_loss.backward()
        self.a_opt.step()

        self.c_opt.zero_grad()
        c_loss.backward()
        self.c_opt.step()


class DDPGAgent:
    def __init__(self, env, input_dim, output_dim, config):
        self.env = env
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.random_process = OUNoise(env.action_space)
        self.config = config

        self.brain = Brain(input_dim, output_dim, config)
        self.memory = Memory(max_size=self.config["max_memory"])

        self.c_mse_loss = nn.MSELoss()

    def replay(self, states, actions, rewards, next_states):

        tpolicy_a_ = self.brain.t_actor.policy(next_states).detach()
        y = rewards + self.config["gamma"] * self.brain.t_critic.qvalue(next_states, tpolicy_a_)

        a_loss = -self.brain.critic.qvalue(states, self.brain.actor.policy(states)).mean()
        c_loss = self.c_mse_loss(self.brain.critic.qvalue(states, actions), y)

        self.brain.update(a_loss, c_loss)
        self.brain.update_targets(tau=self.config["tau"])

        return a_loss, c_loss

    def run_episode(self):
        rewards, a_losses, c_losses = 0, [], []

        self.random_process.reset()
        s = self.env.reset()

        step = 0
        while True:
            a = self.random_process.get_action(self.brain.actor.policy(torch.tensor(s).float()).detach().numpy(), step)

            s_, r, done, _ = self.env.step(a)

            self.memory.store(s, a, r, s_, done)

            if len(self.memory) > self.config["batch_size"]:
                b_states, b_actions, b_rewards, b_next_states, _ = self.memory.sample(self.config["batch_size"])

                a_loss, c_loss = self.replay(b_states, b_actions, b_rewards, b_next_states)

                a_losses.append(a_loss.detach().numpy())
                c_losses.append(c_loss.detach().numpy())

            rewards += r
            s = s_
            step += 1

            if done:
                break

        return rewards, np.mean(a_losses) + np.mean(c_losses)
