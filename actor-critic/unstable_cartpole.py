import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import plot_rwrds_and_aclosses

config = {
    "num_episodes": 2000,
    "max_steps": 501,
    "alpha": 3e-5,
    "beta": 3e-5,
    "gamma": 0.9,
}


class Brain(nn.Module):

    def __init__(self, n_states, n_actions):
        super(Brain, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.actor = nn.Sequential(
            nn.Linear(n_states, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(n_states, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def value(self, states):
        return self.critic(states)

    def action_probs(self, states):
        return F.softmax(self.actor(states))


class A2CAgent:

    def __init__(self, env, input_dim, output_dim):
        self.env = env
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.brain = Brain(input_dim, output_dim)

        self.optimizer = torch.optim.RMSprop(self.brain.parameters(), lr=config["alpha"])

    def compute_returns(self, rewards):
        result = [rewards[-1]]
        for reward in reversed(rewards[:-1]):
            new = reward + config["gamma"] * result[0]
            result.insert(0, new)
        return torch.tensor(result).view(len(result), 1)

    def update(self, s, a, r, s_, entropy_term):
        advantage = r + self.brain.value(s_) - self.brain.value(s)
        a_log_probs = torch.log(self.brain.action_probs(s).gather(0, a))

        a_loss = (-a_log_probs * advantage).mean()
        c_loss = (0.5 * advantage.pow(2)).mean()
        ac_loss = a_loss + c_loss + 0.001 * entropy_term

        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()

        return a_loss, c_loss, ac_loss

    def run_episode(self):

        s = self.env.reset()
        states, actions, rewards, a_losses, c_losses, ac_losses = [], [], [], [], [], []

        entropy_term = 0
        for step in range(config["max_steps"]):
            a_dist = self.brain.action_probs(torch.tensor(s)).detach().numpy()

            a = np.random.choice(self.output_dim, p=a_dist)
            s_, r, done, _ = self.env.step(a)

            entropy_term += -np.sum(np.mean(a_dist) * np.log(a_dist))

            t = torch.tensor
            a_l, c_l, ac_l = self.update(t(s), t(a), t(r), t(s_), entropy_term)

            rewards.append(r)
            actions.append(a)
            states.append(s)
            a_losses.append(a_l.detach().numpy())
            c_losses.append(c_l.detach().numpy())
            ac_losses.append(ac_l.detach().numpy())

            s = s_

            if done:
                break

        return np.sum(rewards), np.mean(a_losses), np.mean(c_losses), np.mean(ac_losses)


def main():
    env = gym.make("CartPole-v1")

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = A2CAgent(env, input_dim, output_dim)

    rewards, a_losses, c_losses, ac_losses = [], [], [], []

    for episode in range(config["num_episodes"]):
        reward, a_loss, c_loss, ac_loss = agent.run_episode()

        rewards.append(reward)
        a_losses.append(a_loss)
        c_losses.append(c_loss)
        ac_losses.append(ac_loss)

        if episode % 10 == 0:
            ac_loss = round(float(ac_loss), 2)
            print("Episode {} \t Reward {} \t Actor Critic Loss {}".format(episode, reward, ac_loss))

    plot_rwrds_and_aclosses(rewards, a_losses, c_losses, ac_losses, roll=50)


if __name__ == '__main__':
    main()
