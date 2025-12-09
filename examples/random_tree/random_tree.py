import numpy as np
import pickle
import sys
import random
import os
from pathlib import Path

from pytens import TensorNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.search import SearchEngine

usage = """
Usage: python random_tree.py <num_samples> <num_repeats> <policy> <seed>

num_samples: int
    Number of samples for MCTS
num_repeats: int
    Number of times to repeat MCTS
policy: UCB1, BUCB1, BUCB2, NormalSampling
    MCTS selection policy
seed: int, default=None
    Random number generator seed
"""


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(usage)
        sys.exit(0)

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

    # Hyperparameters
    config.engine.rollout_max_ops = 0
    config.engine.rollout_rand_max_ops = False
    config.engine.init_num_children = 3
    config.engine.new_child_thresh = 5
    config.engine.explore_param = 1.5

    try:
        # Get the number of samples for MCTS
        num_samples = int(sys.argv[1])

        # Get number of repeats
        num_repeats = int(sys.argv[2])

        # Get required information
        config.engine.policy = sys.argv[3]

    except ValueError:
        print(usage)
        sys.exit(1)

    assert isinstance(num_samples, int)
    assert isinstance(config.engine.policy, str)
    assert isinstance(num_repeats, int)

    # Get random seed
    if len(sys.argv) > 4:
        try:
            seed = int(sys.argv[3])

        except ValueError:
            print(usage)
            sys.exit(1)

        random.seed(seed)
        np.random.seed(seed)

    # Get the number of trees
    dir = Path(f"trees")
    num_trees = len(list(dir.glob("tree_*.pkl")))

    # # Draw MCTS progress
    # config.engine.draw_search = True
    # config.engine.color_by = "mean_score"  # or `state`
    # config.engine.with_labels = False
    # config.engine.filename = "search_mean_score.mp4"
    # config.engine.fps = 3

    # Save data
    partition_data = {
        "Compression Ratio": np.zeros(num_trees),
        "Preprocessing Time": np.zeros(num_trees),
        "Time": np.zeros(num_trees),
        "Count": np.zeros(num_trees),
    }
    mcts_data = {
        "Compression Ratio": np.zeros((num_trees, num_repeats)),
        "Preprocessing Time": np.zeros((num_trees, num_repeats)),
        "Time": np.zeros((num_trees, num_repeats)),
        "Count": np.zeros((num_trees, num_repeats)),
        "TN Indices": np.zeros((num_trees, num_repeats)),
    }

    # Number of trees to run
    for i in range(num_trees):
        print(f"Running Tree {i + 1}")
        with open(f"trees/tree_{i}.pkl", "rb") as f:
            net = TensorNetwork.from_dict(pickle.load(f))

        # Create search engine
        engine = SearchEngine(config)

        # Run partition search
        print("Running Partition Search", end="")
        partition_stats = engine.partition_search(net)
        partition_data["Compression Ratio"][i] = partition_stats["cr_core"]
        partition_data["Preprocessing Time"][i] = partition_stats["preprocess"]
        partition_data["Time"][i] = partition_stats["time"]
        partition_data["Count"][i] = partition_stats["count"]
        print(f", Run Time = {partition_data['Time'][i]} s")

        # Run MCTS
        for j in range(num_repeats):
            print(f"({j}) Running MCTS", end="")
            mcts_stats = engine.mcts(net, num_samples)
            mcts_data["Compression Ratio"][i, j] = mcts_stats["cr_core"]
            mcts_data["Preprocessing Time"][i, j] = mcts_stats["preprocess"]
            mcts_data["Time"][i, j] = mcts_stats["time"]
            mcts_data["Count"][i, j] = mcts_stats["count"]
            mcts_data["TN Indices"][i, j] = mcts_stats["best_tn_idxs"][0]
            print(f", Run Time = {mcts_data['Time'][i, j]} s")

    # Save data
    dir = Path(f"data_{num_trees}trees")
    os.makedirs(dir, exist_ok=True)

    with open(dir / f"partition_{config.engine.policy}.pkl", "wb") as f:
        pickle.dump(partition_data, f)
    with open(dir / f"mcts_{config.engine.policy}.pkl", "wb") as f:
        pickle.dump(mcts_data, f)
