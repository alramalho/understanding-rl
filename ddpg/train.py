from utils import Bcolors, plot_rwrds_and_aclosses
from ddpg_utils import OUNoise
import gym
from agent import DDPGAgent
from gym import wrappers


def train(config):
    print(f'{Bcolors.OKGREEN}Playing problem {config["problem"]}{Bcolors.ENDC}')
    total_rewards, a_losses, c_losses = [], [], []

    env = gym.make(config["problem"])

    if config["save_video"]:
        env = wrappers.RecordVideo(
            env=env,
            video_folder='./video',
            episode_trigger=lambda e: e == config["num_episodes"] - 1,
            name_prefix=f'ddpg-{config["problem"].lower()}'
        )

    random_process = OUNoise(env.action_space)
    agent = DDPGAgent(env, env.observation_space.shape[0], env.action_space.shape[0], random_process, config)

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
