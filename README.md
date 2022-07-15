# Minimal Pytorch RL Algorithms

----

Goals of the repo:
- Providing bridge between readable pseudocode and pytorch implementation to the inexperienced user
  - There lots of great minimal repos ([like this one](https://github.com/seungeunrho/minimalRL)) already. If you just want
- Familiarize myself with implementing algorithms

# TODO - level 0

- put pseudocde image into every folder
- run every algorithm and put some graphs
- solve cartpole with Actor Critic (current stabilizing around 100 reward) ✅

# TODO - level 1
- Add at least 2 environments per algorithm
- unify all `train.py`
- `test` functions
- consistent signature across all algorithms

# TODO - level 2
- separate plotting from main execution (write to intermidiary csv)
- unit tests to verify all algos work
- add logging (tensorboard)
- vectorized environments

# TODO - level 3
- Allow for video saving
- Add Atari onto DQN



# other improvements
- Create `Environment` class with `run_episode` code
- Convert n step actor critic to latex instead of code (also not to be copy paste from other source)
