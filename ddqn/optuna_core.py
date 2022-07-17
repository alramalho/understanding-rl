import optuna
from typing import Callable, Tuple
import numpy as np
from core import create, train


def optuna_create(execution_config, agent_config) -> Tuple[optuna.Study, Callable]:
    def objective(trial: optuna.Trial):
        trials_config = {
            # "net_arch": trial.suggest_categorical("net_arch", ["mlp_small", "mlp_medium"]),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
            "exploration_fraction": trial.suggest_uniform("exploration_fraction", 0, 0.5),
            "target_update_interval": trial.suggest_categorical("target_update_interval",
                                                                [10, 100, 200, 500, 1000, 1500])
        }

        agent = create(execution_config, dict(agent_config, **trials_config))
        results_df = train(execution_config, agent, is_trial=True,
                           trial_number=trial.number)
        ep_rewards = results_df["reward"].values.tolist()

        return np.mean(ep_rewards[-execution_config["n_episodes"]:])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    return study, objective


def optuna_train(study: optuna.Study, objective: Callable, n_trials):
    class StaleStudyCallback:
        def __init__(self, threshold, trial_number, patience):
            self.threshold = threshold
            self.trial_number = trial_number
            self.patience = patience
            self.cb_list = []

        def __call__(self, study: optuna.study, frozen_trial: optuna.trial):
            study.set_user_attr("previous_best_value", study.best_value)

            if frozen_trial.number > self.trial_number:
                previous_best_value = study.user_attrs.get("previous_best_value", None)
                if previous_best_value * study.best_value >= 0:
                    if abs(previous_best_value - study.best_value) < self.threshold:
                        self.cb_list.append(frozen_trial.number)

                        if len(self.cb_list) > self.patience:
                            print('The study stops now...')
                            print("With number", frozen_trial.number, "and value ", frozen_trial.number)
                            print("The prev and curr best values are {} and {}".format(previous_best_value,
                                                                                       study.best_value))
                            study.stop()

    logging_callback = StaleStudyCallback(threshold=1e-5, patience=20, trial_number=20)
    study.optimize(objective, n_trials=n_trials, timeout=600, callbacks=[logging_callback], n_jobs=-1)
