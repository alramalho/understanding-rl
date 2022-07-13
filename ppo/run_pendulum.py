from train import train

config = {
    "problem": "Pendulum-v1",
    "has_continuous_space": True,
    "has_continuous_actions": True,
    "print_freq": 50,
    "n_episodes": 4000,
    "max_steps": 1000,
    "ppo_update_freq": 4,
    "K_epochs": 80,
    "random_seed": 0,  # 0 means absence of random seed
    "action_std": 0.4,
    "action_std_decay_rate": 0.05,
    "min_action_std": 0.1,
    "action_std_decay_freq": 2500,
    "epsilon": 0.2,
    "gamma": 0.99,
    "lr_actor": 3e-3,
    "lr_critic": 1e-3,
    "entropy_reg": 0.01,
    "entropy_decay_freq": 400, # in episodes
}


if __name__ == "__main__":
    train(config)
