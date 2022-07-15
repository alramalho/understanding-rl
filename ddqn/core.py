import optuna
import pandas as pd
from optuna.visualization import plot_intermediate_values
import gym
from agent import DoubleDQNAgent
import numpy as np
import torch
import json
from utils import plot_rwrds_and_losses, Bcolors
from atari_wrappers import AtariWrapper
from logger import Logger

logger = Logger()


def create(execution_config, agent_config):
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
    logger.log_config(agent, agent_config)

    return agent


def train(execution_config, agent):
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


def plot(results_df: pd.DataFrame, config):
    print(f'{Bcolors.OKGREEN}Plotting{Bcolors.ENDC}')
    plot_rwrds_and_losses(
        rewards=results_df["reward"].values.tolist(),
        losses=results_df["loss"].values.tolist(),
        config=config,
        roll=30
    )

#
# def optuna_train(config):
#     class LoggingCallback:
#         def __init__(self, threshold, trial_number, patience):
#             self.threshold = threshold
#             self.trial_number = trial_number
#             self.patience = patience
#             self.cb_list = []
#
#         def __call__(self, study: optuna.study, frozen_trial: optuna.trial):
#             study.set_user_attr("previous_best_value", study.best_value)
#
#             if frozen_trial.number > self.trial_number:
#                 previous_best_value = study.user_attrs.get("previous_best_value", None)
#                 if previous_best_value * study.best_value >= 0:
#                     if abs(previous_best_value - study.best_value) < self.threshold:
#                         self.cb_list.append(frozen_trial.number)
#
#                         if len(self.cb_list) > self.patience:
#                             print('The study stops now...')
#                             print("With number", frozen_trial.number, "and value ", frozen_trial.number)
#                             print("The prev and curr best values are {} and {}".format(previous_best_value,
#                                                                                        study.best_value))
#                             study.stop()
#
#     def objective(trial):
#         env = gym.make(config["problem"])
#         input_dim = env.observation_space.shape[0]
#         output_dim = env.action_space.n
#
#         trials_config = {
#             "net_arch": trial.suggest_categorical("net_arch", ["mlp_small", "mlp_medium"]),
#             "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
#             "exploration_fraction": trial.suggest_uniform("exploration_fraction", 0, 0.5),
#             "target_update_interval": trial.suggest_categorical("target_update_interval", [10, 100, 200])
#         }
#
#         new_config = dict(config, **trials_config)
#
#         agent = DoubleDQNAgent(env, input_dim, output_dim, new_config)
#         ep_rewards, losses, epsilons = [], [], []
#
#         for episode in range(config["n_episodes"]):
#             reward, loss, epsilon = agent.run_episode()
#             ep_rewards.append(reward)
#             losses.append(loss)
#             epsilons.append(epsilon)
#
#             if episode % config["print_freq"] == 0:
#                 print("########################")
#                 print(f"Episode {episode} with config ")
#                 print(json.dumps(trials_config))
#                 r = round(float(np.mean(ep_rewards[-config["print_freq"]:])), 2)
#                 l = round(float(np.mean(losses[-config["print_freq"]:])), 2)
#                 e = round(float(np.mean(epsilons[-config["print_freq"]:])), 2)
#                 print("Average Reward {}, Average Loss: {}, Average Epsilon {}".format(r, l, e))
#
#         return np.mean(ep_rewards[-config["n_episodes"]:])
#
#     study = optuna.create_study(
#         direction="maximize",
#         sampler=optuna.samplers.TPESampler(seed=42),
#         pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
#     )
#
#     logging_callback = LoggingCallback(threshold=1e-5, patience=20, trial_number=20)
#     study.optimize(objective, n_trials=100, timeout=600, callbacks=[logging_callback])
#
#     plot_intermediate_values(study)
