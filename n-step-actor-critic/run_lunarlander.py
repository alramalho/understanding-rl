from train import train

config = {
    "problem": "LunarLander-v2",
    "has_continuous_space": True,
    "has_continuous_actions": False,
    "num_episodes": 2000,
    "max_steps": 501,
    "learning_rate": 0.002,
    "gamma": 0.99,
    "n_steps": 100,
    "print_freq": 20,
    "entropy_reg": 0.01,
    "random_seed": 543,


}

if __name__ == '__main__':
    train(config)
