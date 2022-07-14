from train import train
config = {
    "problem": "HalfCheetah-v2",
    "max_steps": 1000,
    "num_episodes": 100,
    "print_freq": 20,
    "max_memory": 50000,
    "batch_size": 128,
    "gamma": 0.99,
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "tau": 1e-2,
    "save_video": True
}

if __name__ == '__main__':
    train(config)
