# Minimal Pytorch RL Algorithms

----

## Goals of the repo:
- Demystify RL algorithms by providing minimal code implementations and it's accompanying pseudocode
- Serve as support for (insert article URL here)
- Make implementations easy to extend and experiment (via logging, plotting, and hyperparameter tuning)
- Practice implementing algorithms 

### Disclaimer:
- These implementations aren't supposed to be used in research, but for full-transparency learning. \
As so, no testing or saving pre-trained models are provided.

## Usage

For a complete description run
```
pyhton run.py -h
```

### Train
```
python run.py --algo dqn --env CartPole-v1
```
**Outputs**
```
<algorithm>
  └─<experiment>
      ├─config.txt >> Contains agent configuration 
      ├─log.txt >> Stdout output (useful for customization)
      └─results.csv >> CSV of Rewards and Loss
```

- Such files begin to be written as soon as the experiment starts. Hence interrupting via `CTRL`+`C` \ 
will still yield plottable results.
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

### Plot
Used for plotting losses and rewards
```
python run.py --algo ddpg --env Pendulum-v1 --plot experiment_0
```
- `experiment_0` can be ommited and it will use latest experience for that algorithm, for that environment.

## Usage

# TODO
- [ ] put pseudocde image into every folder
- [ ] run every algorithm and put some graphs
- [x] solve cartpole with Actor Critic (current stabilizing around 100 reward)
- [x] Add at least 2 environments per main algorithm
- [x] unify all `train.py`
- [x] consistent signature across all algorithms
- [ ] Add Atari onto DQN
- [x] add logging
  - To csv
- [ ] explain folder structure
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
