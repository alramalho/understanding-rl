from train import train

config = {
    "problem": "PongNoFrameskip-v4",
    "n_episodes": 100,
    "print_freq": 1,
    "random_seed": 42,

    "batch_size": 32,
    "buffer_max_capacity": 5000,

    "gamma": 0.99,
    "net_arch": "conv",  # can be mlp_small, mlp_medium or conv
    "learning_rate": 0.0001,
    # "full" (fully loads params onto target params every tau steps – or "soft", softly updating every step)
    "target_update_interval": 1000,
    "tau": 1,

    "initial_epsilon": 0.8,
    "final_epsilon": 0.01,
    "exploration_fraction": 0.1,  # where end is reached e.g 0.1 -> epsilon becomes final after 10% of training process

}

if __name__ == "__main__":
    train(config)
