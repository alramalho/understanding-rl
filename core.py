import optuna
from typing import Callable, Tuple
import gym
from utils.plot import plot_rwrds_and_losses
from agent import DoubleDQNAgent
import numpy as np
import torch
from utils.utils import Bcolors, sub_dir_count
from utils.atari_wrappers import AtariWrapper
from utils.logger import create_logger
import pandas as pd

def train_agent(execution_config, agent, experiment_number, trial_number=None, is_trial=False):
    logger = create_logger(execution_config, agent, experiment_number, trial_number=trial_number, is_trial=is_trial)

    # Train
    print(f'{Bcolors.OKGREEN}Training{Bcolors.ENDC}')
    try:
        ep_rewards, losses, epsilons = [], [], []
        for episode in range(execution_config["n_episodes"]):
            reward, loss, epsilon = agent.run_episode()

            logger.log_to_results(",".join([str(round(reward, 2)), str(round(loss, 2)), str(round(epsilon, 2))]) + '\n')

            ep_rewards.append(reward)
            losses.append(loss)
            epsilons.append(epsilon)

            if episode % execution_config["print_freq"] == 0:
                r = round(float(np.mean(ep_rewards[-execution_config["print_freq"]:])), 2)
                l = round(float(np.mean(losses[-execution_config["print_freq"]:])), 2)
                e = round(float(np.mean(epsilons[-execution_config["print_freq"]:])), 2)
                print("Episode {}, Average Reward {}, Average Loss: {}, Average Epsilon {}".format(episode, r, l, e))

    except KeyboardInterrupt:
        pass

    return logger.get_results_df()


def create_agent(execution_config, agent_config):
    print(f'{Bcolors.OKGREEN}Creating{Bcolors.ENDC}')

    env = gym.make(execution_config["problem"])
    if execution_config["problem"] == "PongNoFrameskip-v4":
        env = AtariWrapper(env)

    if execution_config["random_seed"]:
        torch.manual_seed(execution_config["random_seed"])
        np.random.seed(execution_config["random_seed"])

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = DoubleDQNAgent(env, input_dim, output_dim, agent_config)

    return agent


def train(execution_config, agent_config):
    agent = create_agent(execution_config, agent_config)
    experiment_number = sub_dir_count(f'/logs/{execution_config["problem"]}')
    train_agent(execution_config, agent, experiment_number)


def optuna_create(execution_config, agent_config) -> Tuple[optuna.Study, Callable]:
    experiment_number = sub_dir_count(f'/logs/{execution_config["problem"]}')
    def objective(trial: optuna.Trial):
        trials_config = {
            # "net_arch": trial.suggest_categorical("net_arch", ["mlp_small", "mlp_medium"]),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
            "exploration_fraction": trial.suggest_uniform("exploration_fraction", 0, 0.5),
            "target_update_interval": trial.suggest_categorical("target_update_interval",
                                                                [10, 100, 200, 500, 1000, 1500])
        }

        agent = create_agent(execution_config, dict(agent_config, **trials_config))
        results_df = train_agent(execution_config, agent, experiment_number,  is_trial=True, trial_number=trial.number)
        ep_rewards = results_df["reward"].values.tolist()

        return np.mean(ep_rewards[-execution_config["n_episodes"]:])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    return study, objective


def optuna_train(study: optuna.Study, objective: Callable, n_trials):
    study.optimize(objective, n_trials=n_trials, timeout=600, n_jobs=-1)


def tune(execution_config, agent_config, n_trials=100):
    study, objective = optuna_create(execution_config, agent_config)
    optuna_train(study, objective, n_trials=n_trials)


def plot_df(results_df: pd.DataFrame, config):
    print(f'{Bcolors.OKGREEN}Plotting{Bcolors.ENDC}')
    plot_rwrds_and_losses(
        rewards=results_df["reward"].values.tolist(),
        losses=results_df["loss"].values.tolist(),
        config=config,
        roll=30
    )


def plot_exp(experiment_results_path: str):
    result_df = pd.read_csv(f"{experiment_results_path}/results.csv")
    plot_df(results_df=result_df, config={"file": experiment_results_path})


def plot(problem, exp=None):
    if exp is None:
        count = sub_dir_count(f"logs/{problem}/")
        exp = f"logs/{problem}/experiment_{count}"

    plot_exp(exp)
