import os
from pathlib import Path

import pickle
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # All policies possible
    # policies = ["UCB1", "BUCB1", "BUCB2", "NormalSampling"]
    policies = ["UCB1"]

    # Plotting
    colors = [
        "#0072B2",
        "#FFA500",
        "#009900",
        "#CC79A7",
        "#56B4E9",
    ]
    markers = ["s", "o", "triangle_down", "triangle_up", "star"]

    # Get the number of trees
    dir = Path(f"trees")
    num_trees = len(list(dir.glob("tree_*.pkl")))

    # Read in the data
    dir = Path(f"data_{num_trees}trees")
    partition_data = pickle.load(open(dir / f"partition_{policies[0]}.pkl", "rb"))
    mcts_data = {
        policy: pickle.load(open(dir / f"partition_{policies[0]}.pkl", "rb"))
        for policy in policies
    }

    # Compute sample means and standard deviations
    for policy in policies:
        for data_name in ["Compression Ratio", "Time", "TN Indices"]:
            mcts_data[policy]["Compression Ratio"] = (
                np.mean(mcts_data[policy]["Compression Ratio"], axis=1),
                np.std(mcts_data[policy]["Compression Ratio"], axis=1),
            )

    for policy in policies:
        for i in range(0, 126, 25):
            plt.axvline(i, color="k", linestyle="-")

        for i, start in enumerate(range(0, 125, 25)):
            mid = start + 12.5
            plt.text(
                mid,
                max(mcts_data["Compression Ratio"][i * 25, (i + 1) * 25 + 1]),
                f"Case {i+1}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.scatter(
            np.arange(num_trees),
            partition_data["Compression Ratio"],
            "-s",
            label="Partition",
        )
        plt.plot(
            np.arange(num_trees),
            mcts_data["Compression Ratio"],
            "-o",
            label=f"MCTS ({policy if policy != 'NormalSampling' else 'Normal Sampling'})",
        )
