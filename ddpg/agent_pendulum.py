import gym
from gym import wrappers
from ddpg_utils import OUNoise
from utils import plot_rwrds_and_aclosses,Bcolors
from collections import deque
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

ft = torch.FloatTensor

config = {
    "num_episodes": 200,
    "print_freq": 20,
    "step_cap": 1600,
    "critic_hidden_layers": [256],
    "actor_hidden_layers": [256],
    "max_memory": 50000,
    "batch_size": 128,
    "gamma": 0.99,
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "tau": 1e-2,
}


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

        return ft(state_batch), ft(action_batch), ft(reward_batch), ft(next_state_batch), ft(done_batch)

    def __len__(self):
        return len(self.buffer)


class Critic(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Critic, self).__init__()
        dynamic_layers = []

        for i in range(len(config["critic_hidden_layers"]) - 1):
            dynamic_layers.append(nn.Linear(config["critic_hidden_layers"][i], config["critic_hidden_layers"][i + 1]))
            dynamic_layers.append(nn.ReLU())
        self.net = nn.Sequential(
            nn.Linear(n_states + n_actions, config["critic_hidden_layers"][0]), nn.ReLU(),
            *dynamic_layers,
            nn.Linear(config["critic_hidden_layers"][-1], 1)
        )

    def qvalue(self, state, action):
        return self.net(torch.cat([state, action], -1))


class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Actor, self).__init__()
        dynamic_layers = []
        self.net = nn.Sequential(
            nn.Linear(n_states, config["actor_hidden_layers"][0]), nn.ReLU(),
            *dynamic_layers,
            nn.Linear(config["actor_hidden_layers"][-1], n_actions), nn.Tanh()
        )

    def policy(self, state):
        return self.net(state)


class Brain:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.actor = Actor(input_dim, output_dim)
        self.t_actor = Actor(input_dim, output_dim)
        self.critic = Critic(input_dim, output_dim)
        self.t_critic = Critic(input_dim, output_dim)

        self.update_targets(tau=1)

        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=config["actor_lr"])
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=config["critic_lr"])

    def update_targets(self, tau=config["tau"]):
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
    def __init__(self, env, input_dim, output_dim, random_process):
        self.env = env
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.random_process = random_process

        self.brain = Brain(input_dim, output_dim)
        self.memory = Memory(max_size=config["max_memory"])

        self.c_mse_loss = nn.MSELoss()

    def replay(self, states, actions, rewards, next_states):

        tpolicy_a_ = self.brain.t_actor.policy(next_states)
        y = rewards + config["gamma"] * self.brain.t_critic.qvalue(next_states,
                                                         tpolicy_a_.detach())  ## how the fuck does this detach fixes?

        # Actor loss
        a_loss = -self.brain.critic.qvalue(states, self.brain.actor.policy(states)).mean()  # maximize Expected return

        # Critic loss
        c_loss = self.c_mse_loss(self.brain.critic.qvalue(states, actions), y)

        # Gradient step
        self.brain.update(a_loss, c_loss)

        # Target nets update
        self.brain.update_targets(tau=config["tau"])

        return a_loss, c_loss

    def run_episode(self, env):
        rewards, a_losses, c_losses = 0, [], []

        self.random_process.reset()
        s = self.env.reset()

        for step in range(config["step_cap"]):
            a = self.random_process.get_action(self.brain.actor.policy(ft(s)).detach().numpy(), step)

            s_, r, done, _ = env.step(a)

            self.memory.store(s, a, r, s_, done)

            if len(self.memory) > config["batch_size"]:
                b_states, b_actions, b_rewards, b_next_states, _ = self.memory.sample(config["batch_size"])

                a_loss, c_loss = self.replay(b_states, b_actions, b_rewards, b_next_states)

                a_losses.append(a_loss.detach().numpy())
                c_losses.append(c_loss.detach().numpy())

            rewards += r

            s = s_

            if done:
                break

        return rewards, np.mean(a_losses), np.mean(c_losses)


problem = 'Pendulum-v1'


def main():
    print(f'{Bcolors.OKGREEN}Playing problem {problem}{Bcolors.ENDC}')
    total_rewards, a_losses, c_losses = [], [], []

    env = gym.make(problem)

    # uncomment if you want to see a video of your agent on its last episode
    # env = wrappers.RecordVideo(env, video_folder='./video', episode_trigger=lambda e: e == config["num_episodes"] - 1,
    #                            name_prefix=f'ddpg-{problem.lower()}')

    random_process = OUNoise(env.action_space)
    agent = DDPGAgent(env, env.observation_space.shape[0], env.action_space.shape[0], random_process)

    for episode in range(config["num_episodes"]):
        reward, a_loss, c_loss = agent.run_episode(env)

        total_rewards.append(reward)
        a_losses.append(a_loss)
        c_losses.append(c_loss)

        if episode % config["print_freq"] == 0:
            print(f'\nEpisode {episode} got reward {reward}, actor loss {a_loss}, critic loss {c_loss}', end=' ')

    plot_rwrds_and_aclosses(
        total_rewards,
        a_losses,
        c_losses,
        ac_losses=None,
        config=dict(config, **{"agent": type(agent).__name__}),
        roll=100
    )


if __name__ == '__main__':
    main()
