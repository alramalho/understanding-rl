# Minimal RL Algorithms

----
# Why?

This repo was built to try and give a clear and non-overwhelming algorithmic practical view
to the unprofessional reinforcement learning practictioner.  
I found the need for it while trying to 
This repo was built with learning and readability in mind. It uses simple OOP principles for 
reponsibility segreggation and the minimal pytorch weird ass vector manipulations possible.

It is supposed to be

# TODO - level 0

- put pseudocde image into every folder
- run every algorithm and put some graphs
- solve cartpole with Actor Critic (current stabilizing around 100 reward)

# TODO - level 1
- Add at least 2 environments per algorithm
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
