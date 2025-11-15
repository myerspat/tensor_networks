import os
import random
from pathlib import Path
from typing import List, Optional

import networkx as nx
import numpy as np
from pydantic import RootModel
from pydantic.dataclasses import dataclass

from pytens import Index, NodeName, Tensor, TensorNetwork


@dataclass
class Node:
    """Class for node representation in benchmarks"""

    name: NodeName
    indices: List[Index]
    value: Optional[str] = None


@dataclass
class Benchmark:
    """Class for benchmark data storage."""

    name: str
    source: str

    nodes: Optional[List[Node]] = None
    value_file: Optional[str] = None

    def to_file(self, dir: Path = Path("benchmarks/random")):
        benchmark_dir = dir / self.name
        if not os.path.exists(benchmark_dir):
            os.makedirs(benchmark_dir)
        with open(f"{benchmark_dir}/random.json", "w+", encoding="utf-8") as json_file:
            json_str = RootModel[Benchmark](self).model_dump_json(indent=4)
            json_file.write(json_str)

    def to_network(self, normalize=False) -> TensorNetwork:
        """Convert a benchmark into a tensor network."""
        assert isinstance(self.nodes, list)

        network = TensorNetwork()

        edges = {}
        for node in self.nodes:
            node_shape = tuple(i.size for i in node.indices)
            if node.value is None:
                node_value = np.random.randn(*node_shape)
            else:
                with open(node.value, "rb") as value_file:
                    node_value = np.load(value_file).astype(np.float32)
                    if normalize:
                        node_value = node_value / np.linalg.norm(node_value)

            network.add_node(node.name, Tensor(node_value, node.indices))

            for ind in node.indices:
                if ind.name not in edges:
                    edges[ind.name] = set()

                edges[ind.name].add(node.name)

        for nodes in edges.values():
            nodes = list(nodes)
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i + 1 :]:
                    network.add_edge(n1, n2)

        return network


def random_tree(benchmark_name, num_of_nodes: int, free_indices: List[Index]):
    length = num_of_nodes - 2
    arr = [0] * length

    # Generate random array
    for i in range(length):
        arr[i] = random.randint(1, length + 1)

    vertex_set = [0] * num_of_nodes

    # Initialize the array of vertices
    for i in range(num_of_nodes):
        vertex_set[i] = 0

    # Number of occurrences of vertex in code
    for i in range(length):
        vertex_set[arr[i] - 1] += 1

    # construct the edge set
    edge_set = []
    j = 0
    # Find the smallest label not present in prufer[].
    for i in range(length):
        for j in range(num_of_nodes):
            # If j+1 is not present in prufer set
            if vertex_set[j] == 0:
                # Remove from Prufer set and print pair.
                vertex_set[j] = -1
                edge_set.append((j + 1, arr[i]))
                vertex_set[arr[i] - 1] -= 1
                break

    j = 0

    # For the last element
    edge_pair = []
    for i in range(num_of_nodes):
        if vertex_set[i] == 0 and j == 0:
            edge_pair.append(i + 1)
            j += 1
        elif vertex_set[i] == 0 and j == 1:
            edge_pair.append(i + 1)
            edge_set.append(edge_pair)
            edge_pair = []

    g = nx.Graph(edge_set)
    # start from a root we collect all free nodes and find a mapping between nodes and free indices
    # each node should have at least two edges
    node_candidates = []
    for n in g.nodes():
        if len(list(nx.neighbors(g, n))) == 1:
            node_candidates.append(n)

    free_assignment = node_candidates + random.choices(
        list(g.nodes), k=len(free_indices) - len(node_candidates)
    )
    random.shuffle(free_assignment)
    nodes = []
    edges = {}
    for n in g.nodes:
        n_name = f"G{n}"
        n_indices = []
        if n in free_assignment:
            for mi, m in enumerate(free_assignment):
                if m == n:
                    n_indices.append(free_indices[mi])

        for gn in nx.neighbors(g, n):
            l = min(gn, n)
            r = max(gn, n)
            if (l, r) not in edges:
                edges[(l, r)] = random.randint(2, 5)
            n_indices.append(Index(f"s_{l}_{r}", edges[(l, r)]))

        nodes.append(Node(n_name, n_indices))

    return Benchmark(benchmark_name, "random", nodes)
