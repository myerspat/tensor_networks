import os
import pickle
import random
import warnings
import optuna

import matplotlib.pyplot as plt
import numpy as np

from pytens.algs import Index, Tensor, TensorNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.search import SearchEngine

# Number of samples for MCTS and how many times to repeat it
num_samples_per_trial = 25
num_repeats_per_trial = 2  # keep this small but still do at least 2 trials to avoid super lucky or unlucky trials skewing results


def get_dataset():
    # Load data and create base network
    psi = np.load("./circle.npy")
    inds = [
        Index(name, size)
        for name, size in zip(
            ["$i_q$", r"$i_{\mu}$", r"$i_\gamma$", r"$i_{\hat{x}}$", r"$i_{\hat{y}}$"],
            psi.shape,
        )
    ]
    net = TensorNetwork()
    net.add_node("T", Tensor(psi, inds))
    return net


def get_base_config():
    config = SearchConfig()

    config.engine.max_ops = 10
    config.engine.eps = 1e-3
    config.engine.timeout = None
    config.engine.verbose = False
    config.synthesizer.action_type = "osplit"
    config.rank_search.error_split_stepsize = 1
    config.rank_search.fit_mode = "topk"
    config.rank_search.k = 1
    config.engine.rollout_max_ops = 0
    config.engine.rollout_rand_max_ops = False

    return config


def objective(trial):
    # Load data and create base network
    net = get_dataset()

    # Setup MCTS config
    config = get_base_config()

    # Hyperparameters
    config.engine.policy = trial.suggest_categorical(
        "policy", ["UCB1", "BUCB1", "BUCB2", "NormalSampling"]
    )
    config.engine.init_num_children = trial.suggest_categorical(
        "initial_children", [2, 3, 4, 5]
    )
    config.engine.new_child_thresh = trial.suggest_categorical(
        "new_child_threshold", [5, 6, 7, 8]
    )
    config.engine.explore_param = trial.suggest_float("C", 0, 5, step=0.1)

    # Run MCTS num_repeats times and average the CR
    engine = SearchEngine(config)

    cr = np.zeros(num_repeats_per_trial)
    for i in range(num_repeats_per_trial):
        cr[i] = engine.mcts(net, num_samples_per_trial)["cr_core"]

    return np.mean(cr)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # # Set random state
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)

    net = get_dataset()
    print("Dataset: {}-dimensional of shape {}".format(len(net.shape()), net.shape()))

    # Run partition for reference
    print("Running Partition")
    engine = SearchEngine(get_base_config())
    stats = engine.partition_search(get_dataset())
    print(
        "CR = {}, Run Time = {} s, Preprocessing Time = {} s, Iteration Count = {}".format(
            *[stats[s] for s in ["cr_core", "time", "preprocess", "count"]]
        )
    )

    # Save the TN
    os.makedirs("networks", exist_ok=True)
    os.makedirs("figs", exist_ok=True)
    with open("networks/partition.pkl", "wb") as f:
        pickle.dump(stats["best_network"].to_dict(), f)

    # Plot the best TN
    plt.clf()
    stats["best_network"].draw()
    plt.savefig("figs/partition.png", dpi=300, transparent=True)

    # Run hyperparameter tuning below
    # Keep n_trials low for debugging, for real tuning make 100-200
    n_trials = 2
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    # optimize over the objective function for a number of trials
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    for key, val in best_trial.params.items():
        print(f"{key}:{val}")

    # write those params to an output file
    with open("circle_tuning.txt", "w") as f:
        f.write(f"Best Params: {best_trial.params} \n")
        f.write(f"Best Values: {best_trial.values}")
