from train import train

config = {
    "problem": "CartPole-v1",
    "n_episodes": 500,
    "print_freq": 10,
    "max_steps": 500,
    "batch_size": 32,
    "buffer_max_capacity": 100000,
    "gamma": 0.99,
    "learning_rate": 0.001,
    # "full" (fully loads params onto target params every tau steps – or "soft", softly updating every step)
    "target_update_interval": 100,
    "tau": 1,
    "init_epsilon": 0.8,
    "min_epsilon": 0.01,
    "epsilon_decay_freq": 50,  # in steps
    "random_seed": 543,
}

if __name__ == "__main__":
    train(config)
