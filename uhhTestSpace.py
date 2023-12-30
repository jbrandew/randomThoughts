import heapq

def min_size_lists_with_index(list_of_lists):
    heap = [(len(lst), lst, index) for index, lst in enumerate(list_of_lists)]
    heapq.heapify(heap)

    # Pop the smallest list
    smallest = heapq.heappop(heap)
    
    # Find all lists with the same size
    min_size = smallest[0]
    min_size_lists = [smallest]
    
    while heap and heap[0][0] == min_size:
        min_size_lists.append(heapq.heappop(heap))

    return min_size_lists

class Node:
    def __init__(self, index):
        self.adjList = set()
        self.index = index

def complement_graph(graph):
    complement_nodes = [Node(node.index) for node in graph]

    for i in range(len(graph)):
        for j in range(len(graph)):
            if i != j:
                node1 = complement_nodes[i]
                node2 = complement_nodes[j]

                # Check if the edge exists in the original graph
                if node2 not in graph[i].adjList and node2 != graph[i]:
                    # Add the complement edge
                    node1.adjList.add(node2)

    return complement_nodes

# Example usage:
node1 = Node(1)
node2 = Node(2)
node3 = Node(3)

node1.adjList = {node2}
node2.adjList = {node1, node3}
node3.adjList = {node2}

graph = [node1, node2, node3]

complement = complement_graph(graph)

for node in complement:
    print(f"Node {node.index} is connected to nodes {[neighbor.index for neighbor in node.adjList]} in the complement graph.")
