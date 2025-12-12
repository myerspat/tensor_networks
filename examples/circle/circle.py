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
num_repeats_per_trial = 10  # keep this small but still do at least 2 trials to avoid super lucky or unlucky trials skewing results


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
    config.engine.init_num_children = trial.suggest_int("initial_children", 1, 5)
    config.engine.new_child_thresh = trial.suggest_int("new_child_threshold", 3, 8)
    config.engine.explore_param = trial.suggest_loguniform("C", 0.00001, 10000)

    # Run MCTS num_repeats times and average the CR
    engine = SearchEngine(config)

    cr = np.zeros(num_repeats_per_trial)
    for i in range(num_repeats_per_trial):
        cr[i] = engine.mcts(net, num_samples_per_trial)["cr_core"]

    return np.mean(cr)


def run_mcts(params_file, num_samples_per_trial=100):
    if params_file is None:
        net = get_dataset()
        # Setup MCTS config
        config = get_base_config()
        # Hyperparameters
        config.engine.policy = "UCB1"
        config.engine.init_num_children = 3
        config.engine.new_child_thresh = 5
        config.engine.explore_param = 1

    else:
        with open(params_file, "r") as f:
            params = eval(f.readlines()[0].replace("Best Params: ", ""))
        net = get_dataset()
        # Setup MCTS config
        config = get_base_config()
        # Hyperparameters
        config.engine.policy = params["policy"]
        config.engine.init_num_children = params["initial_children"]
        config.engine.new_child_thresh = params["new_child_threshold"]
        config.engine.explore_param = params["C"]

    # Run MCTS once and return compression ratio
    engine = SearchEngine(config)

    cr = np.zeros(10)
    best_net = None
    best_cr = 0
    for i in range(10):
        stats = engine.mcts(net, num_samples_per_trial)
        cr[i] = stats["cr_core"]

        if cr[i] > best_cr:
            best_net = stats["best_network"]

    return np.mean(cr), np.std(cr), best_net


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    run_type = "tuning"

    if run_type.lower() not in ["mcts", "partition", "tuning"]:
        raise ValueError("run_type must be one of: mcts, partition, or tuning")
    # # Set random state
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)

    net = get_dataset()
    print("Dataset: {}-dimensional of shape {}".format(len(net.shape()), net.shape()))

    if run_type.lower() == "partition":
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

    elif run_type.lower() == "tuning":
        # Run hyperparameter tuning below
        # Keep n_trials low for debugging, for real tuning make 100-200
        n_trials = 100
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
            f.write(f"Best Avg. Compression Ratio: {best_trial.values}")

    elif run_type.lower() == "mcts":
        tuning_done = os.path.exists("circle_tuning.txt")
        if tuning_done:
            n_iters = [25, 50, 100, 150, 175, 200]
            crs = np.zeros(len(n_iters))
            stds = np.zeros(len(n_iters))

            for i, n in enumerate(n_iters):
                crs[i], stds[i], net = run_mcts(
                    params_file="circle_tuning.txt", num_samples_per_trial=n
                )

                with open(f"networks/best_network_{n}.pkl", "wb") as f:
                    pickle.dump(net.to_dict(), f)

                # Plot the best network
                plt.clf()
                net.draw()
                plt.savefig(f"figs/best_network_{n}.png", dpi=300, transparent=True)

            plt.clf()
            fig, ax = plt.subplots()
            ax.plot(n_iters, crs, marker="o")
            ax.set_xlabel("Number of MCTS Samples")
            ax.set_ylabel("Compression Ratio")
            ax.grid(True)
            plt.savefig("figs/cr_vs_iterations_circle.png", dpi=300)
        else:
            print(
                "Using generic parameters. Please run hyperparameter tuning first for optimal MCTS performance."
            )
            run_mcts(params_file=None)
