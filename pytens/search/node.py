"""Node for Monte Carlo Tree Search"""

from enum import Enum
from itertools import count

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pytens.algs import TensorNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.state import SearchState


def bucb1_score(prior_mean, prior_variance):
    updated_mean = 0
    updated_variance = 1
    return


def bucb2_score(prior_mean, prior_variance):
    updated_mean = 0
    updated_variance = 1
    return


def welford_update(n_visits, mean, M2, new_value):
    """Uses welford's algorithm to update the mean and variance as new samples are taken (online algorithm).

    Args:
        n_visits (int): Number of visits EXCLUDING this one (we're about to increment it)
        mean (float): Mean score at the node before this sample
        M2 (float): Sum of squared distance from the mean before this sample
        new_value (float): 1/size of the tensor network we just rolled out to and need to add to the statistics

    Returns:
        tuple: (n_visits, updated mean, updated variance, updated M2)
    """
    # increment the visits counter
    n_visits += 1
    delta = new_value - mean
    mean += delta / n_visits
    M2 += delta * delta
    # return new mean and variance, plus M2 for tracking
    return n_visits, mean, M2 / n_visits, M2


class NodeState(Enum):
    # This node still has unborn children
    ACTIVE = 0

    # This node has all born children but their children have untried children
    FULLY_BRANCHED = 1

    # This node has no more viable children that can have children
    NO_BRANCH_POSSIBLE = 2


class Node:
    """MCTS node implementation"""

    id_iter = count()
    state2str = {
        NodeState.ACTIVE: "ACTIVE",
        NodeState.FULLY_BRANCHED: "FULLY_BRANCHED",
        NodeState.NO_BRANCH_POSSIBLE: "NO_BRANCH_POSSIBLE",
    }

    def __init__(self, search_state: SearchState, parent=None):
        """
        Initialize MCTS Node.

        Parameters
        ----------
        search_state: pytens.search.state.SearchState
            The search state for the node.
        parent: None or pytens.search.node.Node
            Parent node of this child node.
        """
        self.id = next(self.id_iter)
        self.state = NodeState.ACTIVE

        self.search_state = search_state
        self.parent = parent
        self.children = []
        self.num_potential_children = 0

        self.num_visits = 0
        self.num_visits_since_last_child = -1
        self.mean = (
            0  # this mean will be updated differently depending on the search policy
        )
        # variance is only necessary for bayesian search policies
        self.variance = 0
        # accumulator for the squared distance from the mean size
        self.M2 = 0

    # =================================================================================
    # Node methods

    def traverse(self, config: SearchConfig, depth: int, total_visits: int):
        """
        Selection for MCTS.

        Parameters
        ----------
        config: pytens.search.configuration.SearchConfig
            Search configuration.
        depth: int
            Number of splits thus far.
        total_visits: int
            Number of visits to the root node

        Returns
        -------
        node: pytens.search.node.Node
            Node needing a child.
        """
        assert self.state != NodeState.NO_BRANCH_POSSIBLE

        if config.engine.verbose:
            print(
                "-- Node {}: State = {}, # of Children = {}, # of Visits = {}".format(
                    self.id,
                    self.state2str[self.state],
                    len(self.children),
                    self.num_visits,
                )
            )

        # Check if the node can have children
        if self.state == NodeState.ACTIVE:
            # Check if the node does not have enough children
            if len(self.children) < (
                config.engine.init_num_children
                if isinstance(config.engine.init_num_children, int)
                else config.engine.init_num_children(depth)
            ):
                return self, depth

            # Check if the node has been visited enough to have a child
            if self.num_visits_since_last_child == config.engine.new_child_thresh:
                # Set to -1 since this will be incremented in back propogation
                self.num_visits_since_last_child = -1
                return self, depth

        # Get best child based on score assuming we have viable children that
        # can they themselves have children
        best_score = np.inf
        best_child = None
        for child in self.children:
            if child.state != NodeState.NO_BRANCH_POSSIBLE:
                # this is the UCB1 or sampled score!
                score = child.score_func(config, total_visits)

                if config.engine.verbose:
                    print("--   Child {}, Score = {}".format(child.id, score))

                # compare MCTS path scores
                if score > best_score:
                    best_score = score
                    best_child = child

        if best_child != None:
            return best_child.traverse(config, depth + 1, total_visits)

        # Check if the node is active and allow another branch
        if self.state == NodeState.ACTIVE:
            self.num_visits_since_last_child = -1
        else:
            # This branch has not been labeled to have no more branches/children
            self.state = NodeState.NO_BRANCH_POSSIBLE

        return self, depth

    def backpropagate(self, config: SearchConfig, size):
        """
        Backpropagate score of a child.

        Parameters
        ----------
        config: pytens.search.configuration.SearchConfig
            Search configuration.
        size: float
            The number of elements in the tensor network.
        """
        self.num_visits, self.mean, self.mean_variance, self.M2 = welford_update(
            n_visits=self.num_visits, mean=self.mean, M2=self.M2, new_value=(1 / size)
        )
        self.num_visits_since_last_child += 1

        if config.engine.verbose:
            print(
                "-- Node {}: Mean Score = {}, # of Visits = {}".format(
                    self.id, self.mean, self.num_visits
                )
            )

        # Continue back up the tree
        if self.parent is not None:
            self.parent.backpropagate(config, size)

    def get_next_action(self, config: SearchConfig, depth):
        """
        Get the next action for the node.

        Parameters
        ----------
        config: pytens.search.configuration.SearchConfig
            Search configuration.
        depth: int
            Number of splits thus far.

        Returns
        -------
        action: pytens.search.state.Action or None
            Next action.
        """
        # Check if we're past the max number of splits
        if depth > config.engine.max_ops:
            return None

        # List of all legal actions
        actions = self.search_state.get_legal_actions(
            config.synthesizer.action_type == "osplit"
        )

        # Prune actions based on actions taken by other children
        other_child_actions = [
            child.search_state.past_actions[-1] for child in self.children
        ]
        indices_to_remove = []
        for i, action in enumerate(actions):
            for other_action in other_child_actions:
                if action == other_action:
                    # Add index to remove and remove from pool and break
                    indices_to_remove.append(i)
                    other_child_actions.remove(other_action)
                    break

        for i in reversed(indices_to_remove):
            actions.pop(i)

        # Update number of potential children
        self.num_potential_children = len(actions) - 1

        # Check if we've exhausted all actions
        if len(actions) == 0:
            if (
                len(self.children) == 0
                or np.array(
                    [
                        child.state == NodeState.NO_BRANCH_POSSIBLE
                        for child in self.children
                    ]
                ).all()
            ):
                self.terminate()
            else:
                self.fully_branched()
            return None
        elif self.num_potential_children == 0:
            self.fully_branched()

        # Sample random random action uniformly
        return actions[int(np.random.rand() * len(actions))]

    def score_func(self, config: SearchConfig, total_visits: int):
        """
        Score function for the selection process.

        Parameters
        ----------
        config: pytens.search.configuration.SearchConfig
            Search configuration.
        total_visits: int
            Number of visits to the root node

        Returns
        -------
        score: int
            The score of the node.
        """
        if config.engine.policy == "UCB1":
            return self.mean + config.engine.explore_param * np.sqrt(
                np.log(total_visits) / self.num_visits
            )

        if config.engine.policy == "BUCB1":
            # sample a "mean value" of inverse size from a normal distribution centered around the sample mean with variance modeled as inverse sqrt of visits to the node
            sampled_mean = np.random.normal(self.mean, self.variance)
            return sampled_mean + config.engine.explore_param * np.sqrt(
                np.log(total_visits) / self.num_visits
            )

        if config.engine.policy == "BUCB2":
            # sample a "mean value" of inverse size from a normal distribution centered around the sample mean with sample variance
            sampled_mean = np.random.normal(self.mean, self.variance)
            return sampled_mean + config.engine.explore_param * np.sqrt(
                self.variance * np.log(total_visits)
            )

        if config.engine.policy == "NormalSampling":
            # sample a random of inverse size from a normal distribution centered around the sample mean with sample variance
            return np.random.normal(self.mean, self.variance)

        raise RuntimeError("Only the UCB1 policy is supported")

    def append(self, child):
        """
        Add child to parent's list.

        Parameters
        ----------
        child: pytens.search.node.Node
            Child to append.
        """
        assert isinstance(child, Node)
        self.children.append(child)

    # =================================================================================
    # I / O

    def __str__(self):
        return "Node(\n\tid={},\n\tstate={},\n\tparent_id={},\n\tnum_visits={},".format(
            self.id,
            self.state2str[self.state],
            self.parent.id if self.parent != None else "None",
            self.num_visits,
        ) + "\n\tnum_visits_since_last_child={},\n\tnum_potential_children={},\n\tmean={}\n)".format(
            self.num_visits_since_last_child,
            self.num_potential_children,
            self.mean,
        )

    # =================================================================================
    # Node state change methods

    def fully_branched(self):
        """
        Node has no more direct children.
        """
        # Node cannot have any more direct children
        self.state = NodeState.FULLY_BRANCHED

    def terminate(self):
        """
        Node has no more direct and subsequent branching.
        """
        # Node need not be explored anymore
        self.state = NodeState.NO_BRANCH_POSSIBLE

        # Remove search state except for last action
        if self.parent != None:
            self.search_state.past_actions = [self.search_state.past_actions[-1]]

        # Terminate up the tree too
        if (
            self.parent != None
            and self.parent.num_potential_children == 0
            and np.array(
                [
                    child.state == NodeState.NO_BRANCH_POSSIBLE
                    for child in self.parent.children
                ]
            ).all()
        ):
            self.parent.terminate()

    # =================================================================================
    # Plotting

    def draw(self):
        """
        Draw the node's tensor network.
        """
        plt.clf()
        self.search_state.network.draw()

    def add_node_to_graph(self, G: nx.DiGraph):
        """
        Add the node to the networkx graph.
        """
        G.add_node(id(self), label=str(self.id), state=self.state, mean=self.mean)

        for child in self.children:
            G.add_edge(id(self), id(child))
            child.add_node_to_graph(G)

    # =================================================================================
    # Class methods

    @classmethod
    def create_root(cls, net: TensorNetwork, delta: float):
        """
        Create a root node.
        """
        # Add score to root node
        cls.score = None
        root = cls(
            search_state=SearchState(net, delta),
            parent=None,
        )
        root.score = root.search_state.network.cost()
        # added this because we're keeping track of inverse size so we can maximize
        root.mean = 1 / root.score
        return root
