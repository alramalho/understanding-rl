from train import train

config = {
    "problem": "LunarLander-v2",
    "has_continuous_space": True,
    "has_continuous_actions": False,
    "n_episodes": 2000,
    "max_steps": 501,
    "lr_actor": 0.002,
    "lr_critic": 0.002,
    "gamma": 0.99,
    "print_freq": 20,
    "random_seed": 543,
    "ppo_update_freq": 4,
    "K_epochs": 80,
    "epsilon": 0.1,
    "entropy_reg": 0.01,
    "entropy_decay_freq": 10,  # in episodes
}


if __name__ == '__main__':
    train(config)
