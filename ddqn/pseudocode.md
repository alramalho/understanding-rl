![pseudocode.png](../_imgs/ddqn_pseudocode.png)

On top of that, on the line "Every C steps reset Qhat=Q", you can do a soft update
by providing a `tau` < 1.
