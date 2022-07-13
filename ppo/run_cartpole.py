from train import train

config = {
    "problem": "CartPole-v1",
    "has_continuous_space": True,
    "has_continuous_actions": False,
    "print_freq": 50,
    "n_episodes": 500,
    "max_steps": 1000,
    "ppo_update_freq": 4,
    "K_epochs": 80,
    "random_seed": 0,  # 0 means absence of random seed
    "epsilon": 0.1,
    "gamma": 0.99,
    "lr_actor": 4e-4,
    "lr_critic": 4e-4,
    "entropy_reg": 0.01,
    "entropy_decay_freq": 10,  # in episodes
}

if __name__ == "__main__":
    train(config)
