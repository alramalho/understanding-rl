import optuna
from typing import Callable, Tuple
import gym
from ddqn.agent import DDQNAgent
from ddpg.agent import DDPGAgent
from n_step_actor_critic.agent import NStepActorCriticAgent
from ppo.agent import PPOAgent
from reinforce_with_baseline.agent import ReinforceWithBaselineAgent
from semi_gradient_sarsa.agent import SemiGradientSarsaAgent
import numpy as np
import torch
from _utils.utils import Bcolors, sub_dir_count
from _utils.atari_wrappers import AtariWrapper
from _utils.logger import create_logger

# TRAINING
def train_agent(algo, agent, config, experiment_number, trial_number=None, is_trial=False):
    logger = create_logger(algo, agent, config, experiment_number,
                           trial_number=trial_number, is_trial=is_trial)

    # Train
    try:
        ep_rewards, losses = [], []
        logger.log_to_results("reward,loss\n")
        for episode in range(config["n_episodes"]):
            reward, loss = agent.run_episode()

            logger.log_to_results(",".join([str(round(reward, 2)), str(round(loss, 2))]) + '\n')

            ep_rewards.append(reward)
            losses.append(loss)

            if episode % config["print_freq"] == 0:
                r = round(
                    float(np.mean(ep_rewards[-config["print_freq"]:])), 2)
                l = round(float(np.mean(losses[-config["print_freq"]:])), 2)
                print("Episode {}, Average Reward {}, Average Loss: {}".format(episode, r, l))

    except KeyboardInterrupt:
        pass

    return logger.get_results_df()


def create_env(config):
    env = gym.make(config["problem"])
    if config["problem"] == "PongNoFrameskip-v4":
        env = AtariWrapper(env)

    if config["random_seed"]:
        torch.manual_seed(config["random_seed"])
        np.random.seed(config["random_seed"])
        env.seed(config["random_seed"])

    if config["has_continuous_actions"]:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    if config["has_continuous_space"]:
        space_dim = env.observation_space.shape[0]
    else:
        space_dim = env.observation_space.n

    if config["problem"] == "PongNoFrameskip-v4":
        env = AtariWrapper(env)

    return env, space_dim, action_dim


def create_agent(env, input_dim, output_dim, algo, config):
    if algo == "ddqn":
        agent = DDQNAgent(env, input_dim, output_dim, config)
    elif algo == "ddpg":
        agent = DDPGAgent(env, input_dim, output_dim, config)
    elif algo == "ppo":
        agent = PPOAgent(env, input_dim, output_dim, config)
    elif algo == "n_step_actor_critic":
        agent = NStepActorCriticAgent(env, input_dim, output_dim, config)
    elif algo == "reinforce_with_baseline":
        agent = ReinforceWithBaselineAgent(
            env, input_dim, output_dim, config)
    elif algo == "semi_gradient_sarsa":
        agent = SemiGradientSarsaAgent(env, input_dim, output_dim, config)
    else:
        raise ValueError(f"No agent with name {algo}")
    return agent


def train(algo, config):
    env, input_dim, output_dim = create_env(config)
    agent = create_agent(env, input_dim, output_dim, algo, config)
    experiment_number = sub_dir_count(f'{algo}/logs/{config["problem"]}')
    train_agent(algo, agent, config, experiment_number)


# HYPERPARAMETER TUNING

def optuna_create(algo, config) -> Tuple[optuna.Study, Callable]:
    experiment_number = sub_dir_count(f'{algo}/logs/{config["problem"]}')

    def objective(trial: optuna.Trial):
        trials_config = {
            # only for ddqn
            # "net_arch": trial.suggest_categorical("net_arch", ["mlp_small", "mlp_medium"]),
            # only for n step actor critic
            # "n_steps": trial.suggest_categorical("n_steps", ["10", "50", "100"]),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
            "exploration_fraction": trial.suggest_uniform("exploration_fraction", 0, 0.5),
            "target_update_interval": trial.suggest_categorical("target_update_interval", [10, 100, 200, 500, 1000])
        }

        env, input_dim, output_dim = create_env(config)
        agent = create_agent(env, input_dim, output_dim, algo,
                             dict(config, **trials_config))
        results_df = train_agent(
            algo, agent, config, experiment_number,  is_trial=True, trial_number=trial.number)
        ep_rewards = results_df["reward"].values.tolist()

        return np.mean(ep_rewards)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    return study, objective


def optuna_train(study: optuna.Study, objective: Callable, n_trials):
    try:
        study.optimize(objective, n_trials=n_trials, timeout=600, n_jobs=-1)
    except KeyboardInterrupt:
        print("Study Interrupted")
        pass

    print(f'{Bcolors.OKGREEN}Best Trial was {study.best_trial.number}{Bcolors.ENDC}')


def tune(algo, config, n_trials):
    study, objective = optuna_create(algo, config)
    optuna_train(study, objective, n_trials=n_trials)
