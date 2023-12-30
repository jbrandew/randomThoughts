import math 
from itertools import combinations
import numpy as np 
import pdb 
import heapq

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


def count_subgraphs(graph, subgraphSize):
    """
    Input: 
    graph = graph in the format of index = node, array at index = nodes connected to that node
    k = # size subgraph we are considering 
    
    Output: 
    # of subgraphs with each # of edges 
    """

    #so, we will be iterating over n choose k subgraphs 
    #as we have len(graph) nodes and are examining size k subgraphs 
    numNodes = len(graph)
    iterateNum = math.comb(numNodes, subgraphSize) 
    getIndexCombos = get_index_combinations(numNodes, subgraphSize) 

    ind = 0
    hold = np.ones(len(getIndexCombos))

    #for each set of indices corresponding to a subgraph  
    for indexCombo in getIndexCombos: 
        #now, we have the set of nodes we are considering. 
        #now, create a list of size corresponding to number of nodes
        edgeNumToEachNode = np.zeros(numNodes)
        
        #then, for each node that we are considering an edge from 
        for nodeFrom in indexCombo: 
            #for each node we are considering edge to 
            for nodeToInd in graph[nodeFrom].adjList: 
                edgeNumToEachNode[nodeToInd]+=1 

        #then, get the sum, only examining those who are in the indexCombo 
        sum = 0 
        for nodeFrom in indexCombo: 
            sum+=edgeNumToEachNode[nodeFrom]

        hold[ind] = sum/2 
        ind +=1 

    return np.unique(hold, return_counts=True)     

#not extensively tested 
def two_min_size_lists_with_index(list_of_lists):
    heap = [(len(lst), lst, index) for index, lst in enumerate(list_of_lists)]
    heapq.heapify(heap)

    # Pop the two smallest lists
    smallest = heapq.heappop(heap)
    second_smallest = heapq.heappop(heap)

    return smallest[1], smallest[2], second_smallest[1], second_smallest[2]

class Node(): 

    def __init__(self, index): 
        self.adjList = set() 
        self.index = index 

    def __str__(self):
        return "Node: "+(str(self.index))+" with neighbors: " +str(self.adjList)


def num_common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common_elements = set1.intersection(set2)
    return len(list(common_elements))

def getGraphComplement(graphIn): 
    """
    Get the graph complement. So edge that wasnt there, is now. Edge that is there,
    isnt there now. 

    Inputs: 
    graphIn: graph to get complement of  

    Output: 
    graphOut: complement graph 
    
    """

    #create list of nodes as adjacency list 
    complementGraph = [Node(nodeInd) for nodeInd in range(numNodes)]  
    
    #for each node to return 
    for node in complementGraph: 
        #add adj list as all but ourself
        hold = list(np.arange(len(graphIn)))
        hold.remove(node.index)
        node.adjList = hold

    #then, each node in the new graph, remove the entries from previous
    for node in complementGraph: 
        node.adjList =  [entry for entry in node.adjList if entry not in graphIn[node.index].adjList]

    return complementGraph

def getComplementBalancedGoodAdjList(numNodes): 
    """
    Gets adj list that minimizes clique with max size, and also balances it with the complement. 
    General approach: 
    1. Set up two graphs of nodes. 
    2. Get 2 sets of all possible edges. 
    3. Sort both sets by max # edges in each respective graph.
    4. Sort both sets by # common edges between the two nodes
    5. Add first edge of first set to first graph, remove the added edge from both sets
    6. Add first edge of second set to second graph, remove the added edge from both sets
    7. Repeat steps 3-6 until all edges done 

    Inputs: 
    numNodes: number of nodes in graph 
    
    Outputs: 
    adjList1 and adjList2: adjacency lists of two graphs who have balanced 
    construction :D 
    
    """
    #create list of nodes as adjacency list 
    graph1 = [Node(nodeInd) for nodeInd in range(numNodes)]    
    graph2 = [Node(nodeInd) for nodeInd in range(numNodes)]
    
    #create set of all possible edges to add: 
    #these two sets will mostly be the same, but we make two copies
    #to make sorting faster later on each iteration (as using insertion sort...)
    edgesToAddGraph1 = get_index_combinations(numNodes, 2)
    edgesToAddGraph2 = get_index_combinations(numNodes, 2)

    #iterate while not empty. First set of edges will empty first. 
    #alternating which graph goes first slightly improves results 

    while(len(edgesToAddGraph1) is not 0): 


        #first, order the edges based on max size of adj list based on each pair

        #alternatively...sort them based on the sum of the lengths of the adj list in this graph and the other one as well 
        edgesToAddGraph1 = sorted(edgesToAddGraph1, key=lambda nodePair: max(len(graph1[nodePair[0]].adjList), len(graph1[nodePair[1]].adjList)))
        edgesToAddGraph2 = sorted(edgesToAddGraph2, key=lambda nodePair: max(len(graph2[nodePair[0]].adjList), len(graph2[nodePair[1]].adjList)))
        
        #then, order the edges based on # of common
        
        #would we change something about this when considering the other graph as well? 
        edgesToAddGraph1 = sorted(edgesToAddGraph1, key=lambda nodePair: num_common_elements(graph1[nodePair[0]].adjList, graph1[nodePair[1]].adjList))
        edgesToAddGraph2 = sorted(edgesToAddGraph2, key=lambda nodePair: num_common_elements(graph2[nodePair[0]].adjList, graph2[nodePair[1]].adjList))

        #print( num_common_elements(graph2[edgesToAddGraph1[0][0]].adjList, graph2[edgesToAddGraph1[0][1]].adjList)  )

        #then, take the first edge for graph1 
        edgeToAddGraph1 = edgesToAddGraph1[0] 

        #remove it from possible edges to add 
        edgesToAddGraph1.remove((edgeToAddGraph1))
        edgesToAddGraph2.remove((edgeToAddGraph1))

        #then connect them 
        graph1[edgeToAddGraph1[1]].adjList.add(graph1[edgeToAddGraph1[0]].index)
        graph1[edgeToAddGraph1[0]].adjList.add(graph1[edgeToAddGraph1[1]].index)    

        if(len(edgesToAddGraph2) == 0 or len(edgesToAddGraph1) == 0):
            break

        #repeat process for graph2 
        #then, take the first edge for graph1 
        edgeToAddGraph2 = edgesToAddGraph2[0] 
        #remove it from possible edges to add 
        edgesToAddGraph1.remove((edgeToAddGraph2))
        edgesToAddGraph2.remove((edgeToAddGraph2))

        #then connect them 
        graph2[edgeToAddGraph2[1]].adjList.add(graph2[edgeToAddGraph2[0]].index)
        graph2[edgeToAddGraph2[0]].adjList.add(graph2[edgeToAddGraph2[1]].index)    


    return graph1, graph2 

def getGoodAdjListModified(numNodes, numEdgesToAdd): 
    """
    Gets adj list that mins the clique with max size. General approach: 
    for each edge we are adding: 
    1. get all pairs of possible edges. 
    2. sort pairs by the max # edges between the two nodes
    3. sort pairs by the # common edges between the two nodes
    4. add the first pair to the graph 
    5. repeat steps 1->4 until we have added sufficient edges 
    
    inputs: 
    numNodes: number of nodes in graph 
    numEdgesToAdd: number of edges we need to add 

    outputs: 
    adjList: adjacency list in which max clique size is minimized     
    """
    
    #create list of nodes as adjacency list 
    nodeList = [Node(nodeInd) for nodeInd in range(numNodes)]    

    #create set of all possible edges to add: 
    edgesToAdd = get_index_combinations(numNodes, 2)

    #iterate # times equal to numEdgesToAdd 
    for ind in range(numEdgesToAdd):     
        #TO DO: should use insertion sort instead of pythons time sort

        #first, order the edges based on max size of adj list based on each pair 
        edgesToAdd = sorted(edgesToAdd, key=lambda nodePair: max(len(nodeList[nodePair[0]].adjList), len(nodeList[nodePair[1]].adjList)))

        #then, order the edges based on # of common 
        edgesToAdd = sorted(edgesToAdd, key=lambda nodePair: num_common_elements(nodeList[nodePair[0]].adjList, nodeList[nodePair[1]].adjList))

        #then, take the first edge
        edgeToAdd = edgesToAdd[0] 
        #remove it from possible edges to add 
        edgesToAdd.remove((edgeToAdd))
        #then connect them 
        nodeList[edgeToAdd[1]].adjList.add(nodeList[edgeToAdd[0]].index)
        nodeList[edgeToAdd[0]].adjList.add(nodeList[edgeToAdd[1]].index)           
    
    return nodeList

def getGoodAdjList(numNodes, numEdgesToAdd): 
    """
    This function just returns an adjacency list that minimizes the 
    # of cliques that have the max # of edges within 
    
    inputs: 
    numNodes: number of nodes in graph 
    numEdgesToAdd: number of edges we need to add 

    outputs: 
    adjList: adjacency list in which max clique size is minimized 
    """
    
    #create list of nodes as adjacency list 
    nodeList = [Node(nodeInd) for nodeInd in range(numNodes)]

    #iterate # times equal to numEdgesToAdd 
    for ind in range(numEdgesToAdd): 
        
        #first, order the list of nodes based on size  
        nodeList = sorted(nodeList, key=lambda obj: len(obj.adjList))
        
        madeConnection = False 
        for node2Ind in range(numNodes):

            #break if we have already made the connection 
            if(madeConnection): 
                break  

            for node1Ind in range(node2Ind): 
                #print(str(node2Ind)+" and "+str(node1Ind))
                #pdb.set_trace()
                #if the two nodes are not connected 
                if(not nodeList[node1Ind].index in nodeList[node2Ind].adjList): 
                    
                    #then connect them 
                    nodeList[node2Ind].adjList.add(nodeList[node1Ind].index)
                    nodeList[node1Ind].adjList.add(nodeList[node2Ind].index)
                    madeConnection=True 
                    break 
    
    return nodeList 

#what are we doing? Testing basic cases of ramsey numbers with this. 
numNodes = 17
numSizeSubgraph = 4
numEdgesInClique = numSizeSubgraph*(numSizeSubgraph-1)/2  

# numEdgesRequired = int(np.ceil(numNodes*(numNodes-1)/4))
# goodAdjList = getGoodAdjListModified(numNodes, numEdgesRequired)
# comp = getGraphComplement(goodAdjList)

g1, g2 = getComplementBalancedGoodAdjList(numNodes)

result1 = count_subgraphs(g1, numSizeSubgraph)
print(result1) 

result2 = count_subgraphs(g2, numSizeSubgraph)
print(result2) 

print("Number Edges in Full Clique")
print(numEdgesInClique)

#get the # of edges in each subgraph for each 
#so this gets the number of edges in each subgraph of size 4 
#if one of the subgraphs has # edges = (subgraph size * (subgraph size -1))/2
#then we have a complete clique 


# subgraphDist = count_subgraphs(goodAdjList, 4) 
# subgraphDistComplement = count_subgraphs(comp, 4) 

# print("Original Dist")
# print(subgraphDist)
# print("Complement Dist")
# print(subgraphDistComplement )





# hold = getGoodAdjacencyList(6,5)

# for node in hold:
#     print(node)


# Example usage with k = 3 (number of nodes):
#graph = [[1, 2, 3, 4, 5], [0], [0], [0], [0], [0]]

# graph = [[1,2], [0,4], [0,3], [2,5], [1,5], [3,4]]

# k = 3
# result = count_subgraphs(graph, k)
# print(result) 


