import os
import sys
import pickle
import random

import numpy as np
from benchmark import random_tree
from pytens import Index, TensorNetwork

usage = """
Usage: python generate_random_tree.py <num_trees> <seed>

num_trees: int
    Number of TN trees to generate
seed: int, default=None
    Random number generator seed
"""


def get_ith_random_tree(i, cutoffs=[20, 55, 90]):
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
            [
                Index("I0", 16),
                Index("I1", 18),
                Index("I2", 20),
                Index("I3", 22),
                Index("I4", 14),
            ],
        )
    elif i < cutoffs[2]:
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
            7,
            [
                Index("I0", 16),
                Index("I1", 14),
                Index("I2", 10),
                Index("I3", 16),
                Index("I4", 20),
                Index("I5", 12),
                Index("I6", 8),
            ],
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(usage)
        sys.exit(0)

    # Get the number of trees to generate from the user
    try:
        num_trees = int(sys.argv[1])
    except ValueError:
        print(usage)
        sys.exit(1)

    # Get RNG seed from user
    if len(sys.argv) > 2:
        try:
            seed = int(sys.argv[2])

        except ValueError:
            print(usage)
            sys.exit(1)

        random.seed(seed)
        np.random.seed(seed)

    # Create trees directory if it doesn't already exit
    os.makedirs("trees", exist_ok=True)

    for i in range(num_trees):
        # Get random tree structure
        benchmark = get_ith_random_tree(i)

        # Get network from benchmark
        net = benchmark.to_network()

        # Contract to a single tensor
        core = net.contract()
        net = TensorNetwork()
        net.add_node("G0", core)
        with open(f"trees/tree_{i}.pkl", "wb") as f:
            pickle.dump(net.to_dict(), f)
