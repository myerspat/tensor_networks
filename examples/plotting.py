import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
import os


if __name__ == "__main__":
    trees = 10
    try:
        os.mkdir(f"figs_{trees}trees")
    except FileExistsError:
        print(f"Figs directory already exists.")
    try:
        os.mkdir(f"data_{trees}trees")
    except FileExistsError:
        print(f"Data directory already exists.")
    policies = ["UCB1", "BUCB1", "BUCB2", "NormalSampling"]
    methods = ["partition", "mcts"]
    trials = [1, 2, 3, 4, 5]
    partition_dict = {}
    mcts_dict = {}
    data_dict = {
        "method": [],
        "policies": [],
        "trial": [],
        "comp_ratio": [],
        "time": [],
        "count": [],
        "tree": [],
    }
    # load data dicts
    for method in methods:
        for policy in policies:
            for trial in trials:
                with open(
                    f"data_{trees}trees/{method}_{policy}_{trial}.pkl", "rb"
                ) as f:
                    results = pickle.load(f)
                    # print(f'RESULTS: {results}')
                cr = results["Compression Ratio"]
                time = results["Time"]
                count = results["Count"]
                N = len(cr)

                data_dict["method"] += [method] * N
                data_dict["policies"] += [policy] * N
                data_dict["trial"] += [trial] * N

                data_dict["comp_ratio"] += list(cr)
                data_dict["time"] += list(time)
                data_dict["count"] += list(count)
                data_dict["tree"] += list(np.arange(1, N + 1, 1))

    df = pd.DataFrame(data_dict)

    for p in policies:
        policy_df = df[df["policies"] == p]
        plt.clf()
        ax1 = sns.lineplot(
            policy_df,
            x="tree",
            y="time",
            hue="method",
            #  style='method'
        )
        ax1.set(xlabel="Random Tree []", ylabel="Time [s]")
        ax1.grid(True)
        plt.savefig(f"figs_{trees}trees/time_{p}.png", dpi=300)
        plt.clf()
        ax2 = sns.lineplot(policy_df, x="tree", y="comp_ratio", hue="method")
        ax2.grid(True)
        ax2.set(xlabel="Random Tree []", ylabel="Compression Ratio []")
        plt.savefig(f"figs_{trees}trees/CR_{p}.png", dpi=100)

    # sns.lineplot(df, x="tree", y="time", hue="policies", style="method")
    # plt.savefig("figs_100trees/seaborn.png", dpi=300)

    df.to_csv(f"data_{trees}trees/all_runs.csv")
