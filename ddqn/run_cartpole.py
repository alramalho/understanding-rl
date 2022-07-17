from run_common import run

execution_config = {
    "problem": "CartPole-v1",
    "n_episodes": 100,
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
    run(execution_config, agent_config)
