from ..train import train

config = {
    "problem": "Pendulum-v1",
    "has_continuous_space": True,
    "has_continuous_actions": True,
    "num_episodes": 4000,
    "max_steps": 501,
    "learning_rate": 0.002,
    "gamma": 0.99,
    "n_steps": 100,
    "entropy_reg": 0.01,
    "action_std": 0.4,
    "action_std_decay_rate": 0.05,
    "min_action_std": 0.1,
    "action_std_decay_freq": 2500,
    "random_seed": 543,
    "print_freq": 20,
}

if __name__ == '__main__':
    train(config)
