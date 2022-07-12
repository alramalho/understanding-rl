import gym
import numpy as np
import torch
import torch.nn as nn
import random
from utils import plot_rwrds_and_losses

config = {
    "num_episodes": 1000,
    "max_steps": 200,
    "epsilon": 0.2,
    "gamma": 0.98,
    "learning_rate": 0.005
}

class Brain:

    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, act_dim)
        )

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config["learning_rate"])

    def qvalue(self, state):
        # Pytorch idiosyncrasy – only supports batches as input
        state = state.unsqueeze(0)

        return self.q_net(state).squeeze()

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class SemiGradientSarsaAgent:

    def __init__(self, env, obs_dim, act_dim):
        self.env = env
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.epsilon = config["epsilon"]

        self.brain = Brain(obs_dim, act_dim)

    def select_action(self, state):
        # Epsilon - greedy
        if random.random() <= self.epsilon:
            action = random.choice(range(self.act_dim))
        else:
            action = torch.argmax(self.brain.qvalue(state))
        return int(action)


    def decay_epsilon(self):
        min_eps = 0.05
        reduction = (self.epsilon - min_eps)/(config["num_episodes"] * 0.7)
        self.epsilon = self.epsilon - reduction


    def run_episode(self):
        s = torch.FloatTensor(self.env.reset())
        a = self.select_action(s)

        ep_reward = 0
        losses = []
        for step in range(config["max_steps"]):

            s_, r, done, info = self.env.step(a)
            s_ = torch.FloatTensor(s_)
            ep_reward += r

            a_ = self.select_action(s_)

            if done:
                loss = 0.5*(r - self.brain.qvalue(s)[a])**2
                self.brain.update(loss)
                losses.append(loss.detach().numpy())
                break

            loss = 0.5 * (r + config["gamma"] * self.brain.qvalue(s_)[a_] - self.brain.qvalue(s)[a]) ** 2
            self.brain.update(loss)
            losses.append(loss.detach().numpy())

            s = s_
            a = a_
            self.decay_epsilon()

        return ep_reward, np.mean(losses)



def train_agent():
    task = "CartPole-v0"
    env = gym.make(task)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = SemiGradientSarsaAgent(env, obs_dim, act_dim)

    ep_rewards = []
    ep_losses = []
    for episode in range(1, config["num_episodes"] + 1):
        reward, losses = agent.run_episode()
        ep_rewards.append(reward)
        ep_losses.append(losses)
        if episode % 100 == 0:
            print('Episode {} \t\t Avg Reward {} \t\t Avg loss {}'.format(episode, np.mean(ep_rewards[-100:]), losses))

    plot_rwrds_and_losses(ep_rewards, losses=None, config=config, roll=100)


if __name__ == "__main__":
    train_agent()
