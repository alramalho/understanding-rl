from train import train

main_config = {
    "num_episodes": 1000,
    "print_freq": 10  # in episodes
}

agent_config = {
    "max_steps": 501,
    "alpha": 4e-4,
    "beta": 4e-4,
    "gamma": 0.99,
    "entropy_reg": 0.1
}

config = dict(main_config, **agent_config)


if __name__ == '__main__':
    train()
