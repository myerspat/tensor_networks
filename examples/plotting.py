import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns

def plot_runtime(policy, partition_data, mcts_data):
    # Plot run time
    plt.clf()
    trials = partition_data[policy].keys()
    # avg_partition_time = np.mean([partition_data[trial]["Time"] for trial in trials])
    # partition_error = np.std([partition_data[trial]["Time"] for trial in trials])
    colors = ['red', 'blue', 'green', 'orange', 'black']
    for i, trial in enumerate(trials):
        plt.plot(np.arange(0, len(partition_data[policy][trial]["Time"])), partition_data[policy][trial]["Time"], label=f'Partition Trial {trial}', marker='x', color=colors[i])
        plt.plot(np.arange(0, len(mcts_data[policy][trial]["Time"])), mcts_data[policy][trial]["Time"], label=f'MCTS with {policy} Trial {trial}', marker='o', color=colors[i])
    plt.xlabel("Tree Number")
    plt.ylim(0, 5)
    plt.ylabel("Search Time (s)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"figs/time_{policy}.png", dpi=300)
    return

def plot_runtime2(policy, partition_data, mcts_data):
    # Plot run time
    plt.clf()
    trials = partition_data[policy].keys()
    partition_times = np.array([partition_data[policy][trial]["Time"] for trial in trials])
    mcts_times = np.array([mcts_data[policy][trial]["Time"] for trial in trials])

    print(partition_times.shape)

    avg_partition_time = np.mean(partition_times, axis=0)
    partition_error = np.std(partition_times, axis=0)

    avg_mcts_time = np.mean(mcts_times, axis=0)
    mcts_error = np.std(mcts_times, axis=0)

    plt.plot(np.arange(0, len(partition_data[policy][trial]["Time"])), avg_partition_time, label=f'Partition', marker='x')
    plt.plot(np.arange(0, len(mcts_data[policy][trial]["Time"])), avg_mcts_time, label=f'MCTS with {policy}', marker='o')
    plt.xlabel("Tree Number")
    plt.ylim(0, 5)
    plt.ylabel("Search Time (s)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"figs/time2_{policy}.png", dpi=300)
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
    for trial in range(1, 6):
        plt.scatter(partition_data[policy][trial]["Compression Ratio"], mcts_data[policy][trial]["Compression Ratio"], label=f'Trial {trial}')
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
    methods = ['partition','mcts']
    trials = [1, 2, 3, 4, 5]
    partition_dict ={}
    mcts_dict = {}
    data_dict = {'method':[],
                 'policies':[],
                 'trial':[],
                 'comp_ratio':[],
                 'time':[],
                 'count':[],
                 'tree':[]}
    # load data dicts
    for method in methods:
        for policy in policies:
            for trial in trials:
                with open(f'data/{method}_{policy}_{trial}.pkl', 'rb') as f:
                    results = pickle.load(f)
                    # print(f'RESULTS: {results}')
                cr = results['Compression Ratio']
                time = results['Time']
                count = results['Count']
                N = len(cr)

                data_dict['method'] += [method]*N
                data_dict["policies"] += [policy]*N
                data_dict['trial'] += [trial]*N


                data_dict['comp_ratio'] += list(cr)
                data_dict['time'] += list(time)
                data_dict['count'] += list(count)
                data_dict['tree'] += list(np.arange(1, N+1, 1))
                


    # print(mcts_dict) 
    # print(partition_dict)
    # print(data_dict)
    df = pd.DataFrame(data_dict)
    # print(df)

    # mcts_df = df.loc[df['method']=='mcts']

    for p in policies:
        policy_df = df.loc[df['policies']==p].copy()
        sns.lineplot(
                     policy_df, 
                     x='tree', 
                     y='time', 
                     hue='method',
                    #  style='method'
                     )
        plt.savefig(f'figs/seaborn_{p}.png', dpi=300)


    # sns.lineplot(df, x='tree', y='time', hue='policies', style='method')
    # plt.savefig('figs/seaborn.png', dpi=300)

    df.to_csv("data/all_runs.csv")

    # for policy in policies:
    #     plot_compratio(policy=policy, partition_data=partition_dict, mcts_data=mcts_dict)
    #     plot_runtime2(policy=policy, partition_data=partition_dict, mcts_data=mcts_dict)
        
