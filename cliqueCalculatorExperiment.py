import numpy as np
from itertools import combinations
import math

import numpy as np
from itertools import combinations

def get_index_combinations(n, size):
    """
    Generate all sets of indices of a certain size within the range [0, n-1].
    
    Parameters:
    - n: The range of numbers (0 to n-1).
    - size: The size of the index sets to generate.

    Returns:
    A list of tuples, each representing a set of indices.
    """
    if size < 0 or size > n:
        return []

    index_range = list(range(n))
    index_combinations = list(combinations(index_range, size))
    return index_combinations


def count_subgraphs(graph, subgraph_size):
    num_nodes = len(graph)
    num_combinations = math.comb(num_nodes, subgraph_size)
    index_combinations = get_index_combinations(num_nodes, subgraph_size)

    edge_counts = np.zeros(num_combinations, dtype=int)

    for ind, index_combo in enumerate(index_combinations):
        unique_edges = set()
        for node_from in index_combo:
            for node_to in graph[node_from]:
                if node_to in index_combo:
                    unique_edges.add(tuple(sorted([node_from, node_to])))

        edge_counts[ind] = len(unique_edges)

    unique_counts, counts = np.unique(edge_counts, return_counts=True)
    return unique_counts, counts

# Example usage:
graph = [[1, 2, 3, 4, 5], [0], [0], [0], [0], [0]]
subgraph_size = 3
result = count_subgraphs(graph, subgraph_size)
print(result)

