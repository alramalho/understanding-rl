from optuna_core import optuna_train, optuna_create
from core import plot, plot_exp


def run(execution_config, agent_config):
    # normal train
    # agent = create(execution_config, agent_config)
    # results = train(execution_config, agent)

    # hyperparameter finding (longer train)
    study, objective = optuna_create(execution_config, agent_config)
    optuna_train(study, objective, n_trials=100)

    # plot
    # plot_exp("experiment_14h38-2022.07.17/trial_5")

