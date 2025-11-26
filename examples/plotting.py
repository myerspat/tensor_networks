import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_runtime(policy, partition_data, mcts_data):
    # Plot run time
    plt.clf()
    trials = partition_data.keys()
    # avg_partition_time = np.mean([partition_data[trial]["Time"] for trial in trials])
    # partition_error = np.std([partition_data[trial]["Time"] for trial in trials])
    for trial in trials:
        plt.plot(np.arange(0, len(partition_data[trial]["Time"])), partition_data[trial]["Time"], label=f'Partition Trial {trial}', marker='x')
        plt.plot(np.arange(0, len(mcts_data[trial]["Time"])), mcts_data[trial]["Time"], label=f'MCTS with {policy} Trial {trial}', marker='o')
    plt.xlabel("Search Attempt")
    plt.ylim(0, 1)
    plt.ylabel("Search Time (s)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"figs/time_{policy}.png", dpi=300)
    return

def plot_compratio(policy, partition_data, mcts_data):
    # Plot compression ratio
    plt.clf()
    # min_val = min(
    #     min(mcts_data["Compression Ratio"]), min(partition_data["Compression Ratio"])
    # )
    # max_val = max(
    #     max(mcts_data["Compression Ratio"]), max(partition_data["Compression Ratio"])
    # )
    # plt.plot([min_val, max_val], [min_val, max_val], "k--", label="$y = x$")
    for trial in trials:
        plt.scatter(partition_data[trial]["Compression Ratio"], mcts_data[trial]["Compression Ratio"], label=f'Trial {trial}')
    plt.xlabel("Partition Search Compression Ratio")
    plt.ylabel(f"MCTS with {policy} Compression Ratio")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.savefig(f"figs/cr_{policy}.png", dpi=300)
    return

if __name__ == "__main__":
    policies = ["UCB1", "BUCB1", "BUCB2", "NormalSampling"]
    trials = np.arange(1,6,1)
    partition_dict ={}
    mcts_dict = {}
    for trial in trials:
        partition_files = [f'data/partition_{policy}_{trial}.pkl' for policy in policies]
        mcts_files = [f'data/mcts_{policy}_{trial}.pkl' for policy in policies]
        # load data dicts
        for policy, pfile, mfile in zip(policies, partition_files, mcts_files):
            with open(pfile, 'rb') as file1:
                partition_dict[trial] = pickle.load(file1)
            with open(mfile, 'rb') as file2:
                mcts_dict[trial] = pickle.load(file2)
        
    for policy in policies:
        plot_compratio(policy=policy, partition_data=partition_dict, mcts_data=mcts_dict)
        plot_runtime(policy=policy, partition_data=partition_dict, mcts_data=mcts_dict)
        
