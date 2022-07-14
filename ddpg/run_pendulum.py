from train import train

config = {
    "problem": "Pendulum-v1",
    "num_episodes": 200,
    "print_freq": 20,
    "step_cap": 1600,
    "max_memory": 50000,
    "batch_size": 128,
    "gamma": 0.99,
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "tau": 1e-2,
    "save_video": False
}

if __name__ == '__main__':
    train(config)
