import argparse
import core
import yaml
from _utils.utils import get_last_experiment_name
from _utils.plot import plot
from ddpg.agent import DDPGAgent
from ddqn.agent import DDQNAgent
from ppo.agent import PPOAgent
from n_step_actor_critic.agent import NStepActorCriticAgent
from reinforce_with_baseline.agent import ReinforceWithBaselineAgent
from semi_gradient_sarsa.agent import SemiGradientSarsaAgent

ALGOS = {
    "ddqn": DDQNAgent,
    "ddpg": DDPGAgent,
    "ppo": PPOAgent,
    "n_step_actor_critic": NStepActorCriticAgent,
    "reinforce_with_baseline": ReinforceWithBaselineAgent,
    "semi_gradient_sarsa": SemiGradientSarsaAgent
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="RL Algorithm",
                        type=str, required=True, choices=list(ALGOS.keys()))
    parser.add_argument(
        "--env", type=str, help="environment ID, see README.md for available", required=True)
    parser.add_argument(
        "-o", "--optimize", action="store_true", default=False, help="Run hyperparameters search"
    )
    parser.add_argument(
        "-p", "--plot", type=str, required=False, nargs='?', const=-1, default=None, help="Plot Rewards and Losses for experiment. Takes experiment name (uses last if absent)"
    )
    parser.add_argument(
        "--n-trials",
        help="Number of trials for optimizing hyperparameters. "
        "This applies to each optimization runner, not the entire optimization process.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-d", "--delete", action="store_true", default=False, help="Delete last experiment"
    )
    args = parser.parse_args()

    with open(f"{args.algo}/{args.env}_config.yml", "r") as config:
        config = yaml.safe_load(config)

    if args.optimize:
        core.tune(args.algo, config, n_trials=args.n_trials)
    elif args.delete:
        core.delete_last_experience(args.algo, args.env)
    elif args.plot is not None:
        if args.plot == -1:
            args.plot = get_last_experiment_name(args.algo, args.env)
        plot(args.algo, args.env, args.plot)
    else:
        core.train(args.algo, config)
