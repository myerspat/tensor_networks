import matplotlib.pyplot as plt
import numpy as np
from benchmark import random_tree
import pickle

from pytens import Index, TensorNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.search import SearchEngine


def get_ith_random_tree(i, cutoffs=[25, 50, 75, 100]):
    if i < cutoffs[0]:
        return random_tree(
            f"random_test_{i}",
            4,
            [Index("I0", 16), Index("I1", 18), Index("I2", 20), Index("I3", 22)],
        )
    elif i < cutoffs[1]:
        return random_tree(
            f"random_test_{i}",
            5,
            [Index("I0", 16), Index("I1", 18), Index("I2", 20), Index("I3", 22)],
        )
    elif i < cutoffs[2]:
        return random_tree(
            f"random_test_{i}",
            5,
            [
                Index("I0", 16),
                Index("I1", 18),
                Index("I2", 20),
                Index("I3", 22),
                Index("I4", 14),
            ],
        )
    elif i < cutoffs[3]:
        return random_tree(
            f"random_test_{i}",
            6,
            [
                Index("I0", 16),
                Index("I1", 18),
                Index("I2", 20),
                Index("I3", 22),
                Index("I4", 14),
            ],
        )
    else:
        return random_tree(
            f"random_test_{i}",
            8,
            [
                Index("I0", 16),
                Index("I1", 18),
                Index("I2", 20),
                Index("I3", 22),
                Index("I4", 14),
                Index("I5", 10),
                Index("I6", 32),
            ],
        )


if __name__ == "__main__":
    # # Set random states
    # random.seed(42)
    # np.random.seed(42)

    # Parameters
    num_trees = 125  # Number of random trees to generate and test
    num_samples = 200  # Number of samples for MCTS

    # Create configuration, use the same for both partition and MCTS
    # as the MCTS specific parameters are seperate
    config = SearchConfig()

    config.engine.max_ops = 10
    config.engine.eps = 1e-3
    config.engine.timeout = None
    config.engine.verbose = False
    config.synthesizer.action_type = "osplit"
    config.rank_search.error_split_stepsize = 1
    config.rank_search.fit_mode = "topk"
    config.rank_search.k = 1

    # MCTS specific config params
    config.engine.policy = "UCB1"
    # config.engine.policy = "BUCB1"
    # config.engine.policy = "BUCB2"
    # config.engine.policy = "NormalSampling"
    config.engine.rollout_max_ops = 0
    config.engine.rollout_rand_max_ops = False
    config.engine.init_num_children = 3
    config.engine.new_child_thresh = 5
    config.engine.explore_param = -1.5

    # # Draw MCTS progress
    # config.engine.draw_search = True
    # config.engine.color_by = "mean_score"  # or `state`
    # config.engine.with_labels = False
    # config.engine.filename = "search_mean_score.mp4"
    # config.engine.fps = 3

    # Save data
    partition_data = {
        "Compression Ratio": np.zeros(num_trees),
        "Time": np.zeros(num_trees),
        "Count": np.zeros(num_trees),
    }
    mcts_data = {
        "Compression Ratio": np.zeros(num_trees),
        "Time": np.zeros(num_trees),
        "Count": np.zeros(num_trees),
    }
    with open(f"partition_data_{config.engine.policy}.pkl", "wb") as f1:
        pickle.dump(partition_data, f1)
    with open(f"mcts_data_{config.engine.policy}.pkl", "wb") as f2:
        pickle.dump(mcts_data, f2)

    # Number of trees to run
    for i in range(num_trees):
        # Get random tree structure
        benchmark = get_ith_random_tree(i)

        # Get network from benchmark
        net = benchmark.to_network()
        print(f"Random Tree {i}\n{net}")

        # Contract to a single tensor
        core = net.contract()
        net = TensorNetwork()
        net.add_node("G0", core)

        # Create search engine
        engine = SearchEngine(config)

        # Run partition search
        print("Running Partition Search")
        partition_stats = engine.partition_search(net)
        partition_data["Compression Ratio"][i] = partition_stats["cr_core"]
        partition_data["Time"][i] = partition_stats["time"]
        partition_data["Count"][i] = partition_stats["count"]

        # Run MCTS
        print("Running MCTS")
        mcts_stats = engine.mcts(net, num_samples)
        mcts_data["Compression Ratio"][i] = mcts_stats["cr_core"]
        mcts_data["Time"][i] = mcts_stats["time"]
        mcts_data["Count"][i] = mcts_stats["count"]

    # Plot compression ratio
    plt.clf()
    min_val = min(
        min(mcts_data["Compression Ratio"]), min(partition_data["Compression Ratio"])
    )
    max_val = max(
        max(mcts_data["Compression Ratio"]), max(partition_data["Compression Ratio"])
    )
    plt.plot([min_val, max_val], [min_val, max_val], "k--", label="$y = x$")
    plt.scatter(partition_data["Compression Ratio"], mcts_data["Compression Ratio"])
    plt.xlabel("Partition Search Compression Ratio")
    plt.ylabel("MCT Search Compression Ratio")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.savefig("cr.png", dpi=300)

    # Plot run time
    plt.clf()
    min_val = min(min(mcts_data["Time"]), min(partition_data["Time"]))
    max_val = max(max(mcts_data["Time"]), max(partition_data["Time"]))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", label="$y = x$")
    plt.scatter(partition_data["Time"], mcts_data["Time"])
    plt.xlabel("Partition Search Time (s)")
    plt.ylabel("MCT Search Time (s)")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.savefig("time.png", dpi=300)
