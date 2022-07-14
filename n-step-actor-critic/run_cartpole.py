from train import train

config = {
    "problem": "CartPole-v1",
    "has_continuous_space": True,
    "has_continuous_actions": False,
    "num_episodes": 5000,
    "max_steps": 501,
    "learning_rate": 4e-3,
    "gamma": 0.9,
    "n_steps": 100,
    "entropy_reg": 0.01,
    "print_freq": 20,
    "random_seed": 543, # 0 means no random seed
}


if __name__ == '__main__':
    train(config)
