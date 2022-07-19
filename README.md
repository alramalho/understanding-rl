<figure>
  <img src="_imgs/img.png" style="width: 100%" alt="Minimal Pytorch Reinforcement Learning (RL) algorithms">
</figure>

----

## Goals of the repo:

- Demystify RL algorithms by providing minimal, object-oriented implementations and it's accompanying pseudocode
- Serve as support for (insert article URL here)
- Practice implementing algorithms

### Features:
- OOP code for several algorithms, from the basics (Semi Gradient Sarsa) to state of the art (PPO)
- Understandable and intuitive logger via experience tracking (check usage part)
- Easy to understand hyperparameter finding (< 40 lines of code for all algorithms)

### Disclaimer:

- These implementations aren't supposed to be used in research, but for full-transparency learning. \
  As so, no testing capabilities or pre-trained models are provided.

## Algorithms

| Algorithm                                 | Lines of Code | Verified Environments                                 | 
|-------------------------------------------|---------------|-------------------------------------------------------|
| Semi Gradient Sarsa                       | < 100         | `CartPole-v1`                                         |
| Reinforce with Baseline                   | < 100         | `CartPole-v1`                                         |
| Deep Deterministic Policy Gradient (DDPG) | ~ 150         | `HalfCheetah-v2`     , `Pendulum-v1`                  |
| N Step Actor Critic                       | < 150         | `CartPole-v1`, `LunarLander-v2`                       |
| Double Deep Q Network (DDQN)              | < 200         | `CartPole-v1`, `LunarLander-v2`, `PongNoFrameskip-v4` |
| Proximal Policy Optimization (PPO)        | < 200         | `CartPole-v1`, `LunarLander-v2`, `Pendulum-v1`        |

## Usage

For a complete description run

```
pyhton run.py -h
```

### Train

```
python run.py --algo <algo> --env <env>
```

**Outputs**

```
<algorithm>
  └─logs
    └─<environment> 
      └─<experiment_id>
          ├─config.txt >> Contains agent configuration 
          ├─log.txt >> Stdout output (useful for customization)
          └─results.csv >> CSV of Rewards and Loss
```

For example running 

```
python run.py --algo ddqn --env CartPole-v1
```

will yield 
```
ddqn
  └─logs
    └─CartPole-v1 
      └─experiment_0
          ├─config.txt >> Contains agent configuration 
          ├─log.txt >> Stdout output (useful for customization)
          └─results.csv >> CSV of Rewards and Loss
```

- Running again does not overwrite, but appends new experiments `<algo>/logs/<env>` folder
- The files begin to be written as soon as the experiment starts. Hence interrupting via `CTRL`+`C` will still yield plottable results.
- You can also delete last experiment by running the same command with `-d` or `--delete` flag

### Tune / Optimize

This is mostly helpful if you plan on adding different environments.
It uses [optuna](https://optuna.org/) to run several hyperparameter combinations and picks the best.

```
python run.py --algo ddpg --env Pendulum-v1 --optimize --n-trials 100
```

**Outputs**

```
<algorithm>
  └─logs
    └─<environment>
      └─<experiment>
        └─trial_1
          ├─config.txt >> Contains agent configuration 
          ├─log.txt >> Stdout output (useful for customization)
          └─results.csv >> CSV of Rewards and Loss
        └─trial_2
          └─...
        └─...
        └─trial_<n_trials>
          └─...
```

- You can cancel it with CTRL+C
- For simplicity, all hyperparameter _suggestions_ are done in `core.optuna_create` method. I'll leave the tweaking around for you.

### Plot

Used for plotting losses and rewards

```
python run.py --algo <algo> --env <env> --plot <experiment>
```
Ex: `python run.py --algo ppo --env CarPole-v1 --plot experiment_6`

Would open
<figure>
  <img src="_imgs/ppo_cartpole_results.png" style="width: 100%" alt="Example Plotting of PPO results in CartPole-v1">
</figure>



- `<experiment>` can be ommited and it will use latest experience for specified algorithm and environment.


# TODO

- [x] put pseudocde image into every folder
- [ ] run every algorithm and put some graphs
- [x] Add at least 2 environments per main algorithm
- [x] unify all `train.py`
- [x] consistent signature across all algorithms
- [x] Add Atari onto DQN
- [x] add logging
    - To csv
- [x] explain folder structure
- [x] explain experiment
- [x] add no test disclaimer
- [x] separate plotting from main execution (use intermidiary csv)
- [ ] vectorized environments. (Check `stable_baselines3/common/vec_env`)
- [ ] unit tests to verify all algos work (take a look at `rl-baselines3-zoo/tests`)
- [ ] Allow for video saving

# other improvements

- Create `Environment` class with `run_episode` code
- Convert n step actor critic to latex instead of code (also not to be copy paste from other source)

# troubleshooting

- In case you run into ROM license troubles when running `PongNoFrameskip-v4`, run

```
pip install "gym[atari,accept-rom-license]"
```

Be aware that this accepts the ROM license for you.
