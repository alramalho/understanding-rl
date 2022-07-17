from optuna_core import optuna_train, optuna_create, plot

execution_config = {
    "problem": "LunarLander-v2",
    "n_episodes": 500,
    "print_freq": 10,
    "max_steps": 500,
    "random_seed": 543,
}

agent_config = {
    "n_episodes" : execution_config["n_episodes"],
    "batch_size": 32,
    "buffer_max_capacity": 100000,

    "gamma": 0.99,
    "net_arch": "mlp_medium",  # can be mlp_small, mlp_medium or conv
    "learning_rate": 0.001,
    # "full" (fully loads params onto target params every tau steps – or "soft", softly updating every step)
    "target_update_interval": 200,
    "tau": 1,

    "initial_epsilon": 0.8,
    "final_epsilon": 0.01,
    "exploration_fraction": 0.25,  # where end is reached e.g 0.1 -> epsilon becomes final after 10% of the training process

}

if __name__ == "__main__":
    # agent = create(execution_config, agent_config)
    # results = train(execution_config, agent)
    # plot(results, agent.config)

    study, objective = optuna_create(execution_config, agent_config)
    optuna_train(study, objective, n_trials=100)
