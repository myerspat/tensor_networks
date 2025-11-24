"""Monte Carlo Search of tensor network structures."""

import atexit
import copy
import time
from typing import List, Literal, Tuple

import imageio.v2 as imageio
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pytens.algs import TensorNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.constraint import BAD_SCORE
from pytens.search.node import Node, NodeState
from pytens.search.partition import PartitionSearch
from pytens.search.state import Action, SearchState
from pytens.search.utils import remove_temp_dir


class MCTSearch(PartitionSearch):
    """
    Monte Carlo Tree Search class.
    """

    def __init__(self, config: SearchConfig):
        """
        Initialize MCTSearch class.

        Parameters
        ----------
        config: SearchConfig
            Search settings. All MCTS parameters are in ``config.engine``.
        """
        super().__init__(config)
        self.root = None
        if config.engine.draw_search:
            self.fig, self.ax = plt.subplots()
            self.cbar = None

        self.frames = []

    # =================================================================================
    # MCTS methods

    def search(self, net: TensorNetwork, num_samples: int):
        """
        Run MCTS.

        Parameters
        ----------
        net: TensorNetwork
            The target tensor network to optimize.
        num_samples: int
            The number of new tensor network sketches.

        Returns
        -------
        stats: dict
            Dictionary of information with ``"topk"`` tensor networks.
        """
        if self.config.rank_search.fit_mode != "topk":
            raise RuntimeError("Only `topk` is supported")

        self.stats["best_network"] = net
        self.stats["count"] = 0
        delta = net.norm() * self.config.engine.eps
        self.delta = delta
        free_indices = net.free_indices()

        # Root node for MCTS
        if self.root == None:
            self.root = Node.create_root(net, delta)

        # Calculate all SVDs for all possible first splits
        start = time.time()
        self.constraint_engine.preprocess(
            net.contract(),
            compute_uv=self.config.rank_search.fit_mode == "all",
        )
        if self.config.output.remove_temp_after_run:
            atexit.register(
                remove_temp_dir,
                self.config.output.output_dir,
                self.constraint_engine.temp_files,
            )
        toc1 = time.time()

        # Run MCTS
        self.stats["tic"] = time.time()
        # root.mean is 1/size
        best_cost = [1 / self.root.mean]

        if self.config.engine.verbose:
            print("Running MCTS")

        while (
            self.stats["count"] < num_samples
            and self.root.state != NodeState.NO_BRANCH_POSSIBLE
            and (
                self.config.engine.timeout is None
                or (time.time() - start) < self.config.engine.timeout
            )
        ):
            if self.config.engine.verbose:
                assert self.root.score != None
                print(80 * "=")
                print(
                    "Sample {}, Elapsed Time = {}, Best CR = {}".format(
                        self.stats["count"] + 1,
                        time.time() - start,
                        self.root.score / best_cost[0],
                    )
                )
                print("Selection")

            # Selection: traverse MCTS tree until we get to a node that wants
            # a new child
            node, depth = self.root.traverse(self.config, 0, self.stats["count"])

            if self.config.engine.verbose:
                print(f"Selected Node {node.id} at depth = {depth}")

            # Check if node has no more branches below
            if node.state == NodeState.NO_BRANCH_POSSIBLE:
                continue

            # Checks
            assert depth <= self.config.engine.max_ops
            assert len(node.search_state.past_actions) == depth

            # Determine the next split
            action = node.get_next_action(self.config, depth + 1)

            if action != None:
                # Expansion: create new leaf node that's never been visited
                child = Node(
                    search_state=self.pseudo_action_execution(
                        node.search_state, action
                    ),
                    parent=node,
                )
                depth += 1

                # Rollout or simulate: run any further splits defined by the user and
                # assign ranks
                state = (
                    self.rollout(child, depth)
                    if self.config.engine.rollout_rand_max_ops > 0
                    and depth != self.config.engine.max_ops
                    else child.search_state
                )

                # Assign the score to the child
                best_cost, size = self.get_cost(state, best_cost)

                if self.config.engine.verbose:
                    print("Backpropagate")

                # Back propagate
                if size != BAD_SCORE:
                    child.backpropagate(self.config, size)
                else:
                    # Rank optimization failed for some reason, back propogate
                    # a 'neutral' score but keep the node, also make it more
                    # likely to branch on its parent
                    child.backpropagate(self.config, self.root.score)
                    node.num_visits_since_last_child += 1

                # Terminate children at max depth
                if depth == self.config.engine.max_ops:
                    child.terminate()

                # Print child and score
                if self.config.engine.verbose:
                    print("New Child: Score = {}\n{}".format(size, child))

                # Append child to node
                node.append(child)

                # Increment the number of samples
                self.stats["count"] += 1

            if self.config.engine.draw_search:
                # Draw progress
                self.fig, self.ax = self.draw(
                    color_by=self.config.engine.color_by,
                    with_labels=self.config.engine.with_labels,
                    ax=self.ax,
                )

                # Add to movie
                self.fig.canvas.draw()
                frame = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
                frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
                frame = frame[:, :, [1, 2, 3, 0]]  # convert ARGB -> RGBA
                self.frames.append(frame)
                if self.config.engine.color_by == "state":
                    self.legend.remove()
                self.ax.clear()

        if self.config.engine.draw_search:
            self.save_movie(
                filename=self.config.engine.filename, fps=self.config.engine.fps
            )

        # Post processing
        # Realize top candidates
        if self.config.rank_search.fit_mode == "topk":
            # get the smallest and
            # replay with error splits around the estimated ranks
            costs = sorted([(v, k) for k, v in self.costs.items()])
            for _, acs in costs[: self.config.rank_search.k]:
                for k, ac in enumerate(acs):
                    ac.target_size = self.ranks[acs][k]

                self.stats["best_acs"] = acs
                self.replay(self.root.search_state, acs, True)

        toc2 = time.time()

        self.stats["time"] = toc2 - start
        self.stats["preprocess"] = toc1 - start
        self.stats["cr_core"] = (
            float(np.prod([i.size for i in free_indices]))
            / self.stats["best_network"].cost()
        )
        self.stats["cr_start"] = net.cost() / self.stats["best_network"].cost()
        best_tensor = self.stats["best_network"].contract()
        best_tensor_indices = best_tensor.indices
        perm = [best_tensor_indices.index(ind) for ind in free_indices]
        best_tensor = best_tensor.permute(perm)
        self.stats["reconstruction_error"] = float(
            np.linalg.norm(best_tensor.value - net.contract().value)
            / np.linalg.norm(net.contract().value)
        )

        if self.config.engine.verbose:
            print(
                "Finished! # of Iterations = {}, Elapsed Time = {} s, Best CR = {}".format(
                    self.stats["count"], self.stats["time"], self.stats["cr_start"]
                )
            )

        return self.stats

    def rollout(self, child: Node, depth: int):
        """
        MCTS rollout process. Add a number of actions after the newest child node.

        Parameters
        ----------
        child: pytens.search.node.Node
            The child node we're on.
        depth: int
            Number of splits thus far.

        Returns
        -------
        search_state: pytens.search.state.SearchState
            The newest search state that should be evaluated.
        """
        # Get the maximum number of splits
        num_actions = (
            int(np.random.rand() * (self.config.engine.rollout_max_ops + 1))
            if self.config.engine.rollout_rand_max_ops
            else self.config.engine.rollout_max_ops
        )
        num_actions = min(num_actions, self.config.engine.max_ops - num_actions)

        for _ in range(num_actions):
            # Get the next action
            action = child.get_next_action(self.config, depth + 1)

            # Check if we've exhausted all actions
            if action == None:
                break

            # Create the next "child"
            child = Node(
                search_state=self.pseudo_action_execution(child.search_state, action)
            )
            print(child.search_state.past_actions)
            depth += 1

        return child.search_state

    # =================================================================================

    def get_cost(
        self,
        state: SearchState,
        best_cost: List[int],
    ) -> Tuple[List[int], int]:
        """Call a constraint solver to estimate the cost of a given network"""
        if self.config.rank_search.fit_mode == "topk":
            # Run constraint solver to find rank
            rank, cost = self.constraint_engine.get_cost(state, BAD_SCORE)
            if cost != BAD_SCORE:
                best_cost.append(cost)
                best_cost = sorted(best_cost)
                if len(best_cost) > self.config.rank_search.k:
                    best_cost = best_cost[: self.config.rank_search.k]
            self.costs[tuple(state.past_actions)] = cost
            self.ranks[tuple(state.past_actions)] = rank

            return best_cost, cost

        else:
            raise NotImplemented("Only `topk` is supported")

    def replay(
        self,
        st: SearchState,
        actions: List[Action],
        first_iter=False,
    ):
        """Apply the given actions around the given ranks."""
        if not actions:
            for n in st.network.network.nodes:
                net = copy.deepcopy(st.network)
                net.round(n, st.curr_delta)
                if net.cost() < self.stats["best_network"].cost():
                    self.stats["best_network"] = net

            return

        ac = actions[0]
        if first_iter and self.config.rank_search.fit_mode == "all":
            svd_file = self.constraint_engine.first_steps.get(ac, None)
            svd_data = np.load(svd_file)
            svd = (svd_data["u"], svd_data["s"], svd_data["v"])
        else:
            svd = None
        for new_st in st.take_action(ac, svd=svd, config=self.config):
            self.stats["compression"].append(
                (
                    time.time() - self.stats["tic"],
                    new_st.network.cost(),
                )
            )
            ukey = new_st.network.canonical_structure()
            self.stats["unique"][ukey] = self.stats["unique"].get(ukey, 0) + 1
            self.replay(new_st, actions[1:])

    # =================================================================================
    # Drawing

    def draw(
        self,
        color_by: Literal["state", "mean"] = "state",
        with_labels: bool = True,
        ax=None,
    ):
        """
        Plot MCTS.

        Parameters
        ----------
        color_by: "state" or "mean"
            What to color the nodes by.
        with_labels: bool
            Whether to include the node ID labels.
        ax: None or matplotlib.pyplot.axis.Axis
            Axis to plot to.

        Returns
        -------
        fig: matplotlib.pyplot.figure.Figure
            Figure
        ax: matplotlib.pyplot.axis.Axis
            Axis of figure.
        """
        assert self.root != None

        STATE_COLORS = {
            NodeState.ACTIVE: "lightblue",
            NodeState.FULLY_BRANCHED: "lightcoral",
            NodeState.NO_BRANCH_POSSIBLE: "lightgray",
        }

        # Create networkx graph
        G = nx.DiGraph()
        self.root.add_node_to_graph(G)

        # Create tree
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

        # Colors and labels
        node_colors = (
            [STATE_COLORS[G.nodes[n]["state"]] for n in G.nodes]
            if color_by == "state"
            else self.root.score / np.array([G.nodes[n]["mean"] for n in G.nodes])
        )
        node_labels = {n: G.nodes[n]["label"] for n in G.nodes}

        if ax == None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        nx.draw(
            G,
            pos,
            labels=node_labels,
            with_labels=with_labels,
            node_color=node_colors,
            node_size=500,
            font_size=8,
            cmap=None if color_by == "state" else plt.cm.plasma,
            arrows=False,
            ax=ax,
        )
        if color_by == "mean":
            if self.cbar == None:
                self.cbar = fig.colorbar(
                    plt.cm.ScalarMappable(
                        norm=colors.LogNorm(
                            vmin=min(node_colors), vmax=max(node_colors)
                        ),
                        cmap=plt.cm.plasma,
                    ),
                    ax=ax,
                    label="Compression Ratio",
                )
            else:
                self.cbar.update_normal(
                    plt.cm.ScalarMappable(
                        norm=colors.LogNorm(
                            vmin=min(node_colors), vmax=max(node_colors)
                        ),
                        cmap=plt.cm.plasma,
                    )
                )

        else:
            legend_labels = {
                "Active": "lightblue",
                "Fully Branched": "lightcoral",
                "No Branch Possible": "lightgray",
            }

            patches = [
                mpatches.Patch(color=color, label=label)
                for label, color in legend_labels.items()
            ]
            self.legend = fig.legend(handles=patches)

        return fig, ax

    def save_movie(self, filename="search.mp4", fps=5):
        """
        Save video of MCTS progression.

        Parameters
        ----------
        filename: str
            Where to save the video.
        fps: int
            Frames per second.
        """
        imageio.mimsave(filename, self.frames, fps=fps)
