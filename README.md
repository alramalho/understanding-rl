# Minimal Pytorch RL Algorithms

----

Goals of the repo:
- Demystify RL algorithms by providing minimal code implementations and it's accompanying pseudocode
- Serve as support for (insert article URL here)
- Make implementations easy to extend and experiment (via logging, plotting, and hyperparameter tyning)
- Practice implementing algorithms 

Disclaimers:
- It is meant to be fast (run < 5min every algorithm) so some instability due to high learning rates is

# TODO - level -1 (debug)
- Run through `python3 run_<env>` (**FAILING**)

# TODO - level 0

- put pseudocde image into every folder
- run every algorithm and put some graphs
- solve cartpole with Actor Critic (current stabilizing around 100 reward) ✅

# TODO - level 1
- Add at least 2 environments per algorithm
- unify all `train.py`
- `test` functions (take a look at `rl-baselines3-zoo/tests`)
- consistent signature across all algorithms
- Add Atari onto DQN

# TODO - level 2
- add logging 
  - To csv
- explain folder structure
- separate plotting from main execution (use intermidiary csv)

# TODO - level 3
- vectorized environments. (Check `stable_baselines3/common/vec_env`)
- unit tests to verify all algos work
- Allow for video saving



# other improvements
- Create `Environment` class with `run_episode` code
- Convert n step actor critic to latex instead of code (also not to be copy paste from other source)


# troubleshooting

- In case you run into ROM license troubles, run
```
pip install "gym[atari,accept-rom-license]"
```
Be aware that this accepts the ROM license for you.
