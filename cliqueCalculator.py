import math 
from itertools import combinations
import numpy as np 
import pdb 
import heapq

def get_index_combinations(n, size):
    """
    Generate all sets of indices of a certain size within the range [0, n-1].
    Auto sorted by second index, then first index 

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


def create_nodes(adjacency_lists):
    nodes = []
    
    for index, adj_list in enumerate(adjacency_lists):
        node = Node(index)
        node.adjList.update(adj_list)
        nodes.append(node)
    
    return nodes

def countSubgraphsForOneNode(graph, subgraphsize, nodeInd): 
    """
    This method gets the # of subgraphs of each number of edges a certain node
    belongs to (i.e 0 subgraphs with 0 edges, 3 subgraphs with 1 edge, so on)

    Actually, dont implement this. just modify the count_subgraphs method to 
    return this list that we want. Much faster than computing all of them individually. 

    Input: 
    graph: list of "nodes," the object we usually use 
    subgraphsize: the num nodes in each subgraph we are examining 
    nodeInd: the index of the node we are examining here 

    Output: 


    """

    return 

def count_subgraphsSlightlyUpdated(graph, subgraphSize): 
    """
    This is the count_subgraph function with a couple slight changes. 
    This takes in a list of nodes instead. Also, instead of just returning
    the # of each type of subgraph (type = # of edges), it will return a 
    graphSize x (subgraphsize * (subgraphsize -1)/2) size matrix. 

    This matrix represents the # of each type of graph a certain node belongs to. 
    
    Its mostly the same process as the other function though. Corrected a couple
    bugs from previous method to this one though. 

    This function is somewhat well tested. 

    Process: 
    1. Sort graph of nodes based on index
    2. Create storage for output
    3. Repeat count_subgraphs, but instead 
    
    Input: 
    graph: list of node objects
    subgraphsize = size of subgraph we are examining 

    Output: 
    for each node, gives the # subgraphs of each type they are a part of in
    the form of a graphSize x (subgraphsize * (subgraphsize -1)/2) size matrix. 
    
    """
    #1. Sort list of nodes based on index 
    graph = sorted(graph, key = lambda node: node.index)

    #2. Create storage for output 
    numEdgesOfSubgraph = int(subgraphSize*(subgraphSize - 1)/2) 
    numNodes = len(graph) 
    #do + 1 to account for 0 edge case 
    outputCounts = np.zeros([numNodes, numEdgesOfSubgraph+1])
    #3. Repeat previous count subgraphs algorithm with slight adjustment  
    #so, first get the set of subgraphs, as in the set of sets of nodes
    #of the subgraphSize we are examining   
    getIndexCombos = get_index_combinations(numNodes, subgraphSize) 
    #pdb.set_trace() 
    #for each set of indices corresponding to a subgraph  
    for subgraphSetOfNodes in getIndexCombos: 
        numEdges = 0 

        #then, for each node that we are considering an edge from 
        for nodeFrom in subgraphSetOfNodes: 
            #for each node we are considering edge to 
            for nodeToInd in graph[nodeFrom].adjList:    
                if(nodeToInd in subgraphSetOfNodes): 
                    numEdges+=1 
                if(nodeToInd == graph[nodeFrom].index):
                    raise Exception("Your own index shouldnt be in your adj list")

        if(numEdges % 2 != 0): 
            raise Exception("Should of have had even num edges, as counting both directions")
        
        #so, after we get what type of subgraph this is, 
        #increment each entry in output counts accordingly 
        numEdges = int(numEdges/2)
        #
        #pdb.set_trace() 
        for nodeFrom in subgraphSetOfNodes: 
            #print("Hello")
            outputCounts[graph[nodeFrom].index, numEdges]+=1 

    return outputCounts


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
    #sort the nodes first for easy operation 
    graph = sorted(graph, key = lambda node: node.index)

    iterateNum = math.comb(numNodes, subgraphSize) 
    getIndexCombos = get_index_combinations(numNodes, subgraphSize) 

    ind = 0

    #get space for each counting each subgraph 
    hold = np.ones(len(getIndexCombos))

    #for each set of indices corresponding to a subgraph  
    for indexCombo in getIndexCombos: 
        #now, we have the set of nodes we are considering. 
        #now, create a list of size corresponding to number of nodes
        
        #modifying this function slightly. #1 denotes previous version code 
        #1. edgeNumToEachNode = np.zeros(numNodes)
        #2. 
        sum = 0 

        #so this loop is basically getting the distribution of edges
        #going out 

        #then, for each node that we are considering an edge from 
        for nodeFrom in indexCombo: 
            #for each node we are considering edge to 
            for nodeToInd in graph[nodeFrom].adjList: 
                #2. 
                if(nodeToInd in indexCombo): 
                    sum+=1 

               #1. edgeNumToEachNode[nodeToInd]+=1 

        #then, get the sum, only examining those who are in the indexCombo 
        #so, only count the edges that involve those in our subgraph 
        
        #1. 
        #sum = 0 
        #for nodeFrom in indexCombo: 
        #    sum+=edgeNumToEachNode[nodeFrom]
        #1. 

        #divide by 2 as we dont want to count bidirectional 
        hold[ind] = sum/2 
        ind +=1 

    #get the unique values 
    subgraphValues, subgraphCounts = np.unique(hold, return_counts=True) 
    #convert them to lists 
    subgraphValues = subgraphValues.tolist() 
    subgraphCounts = subgraphCounts.tolist() 

    length = int(subgraphSize*(subgraphSize-1)/2)
    #return item 
    returnValues = np.arange(length+1)
    returnCounts = np.zeros(length+1)

    #just filling in gaps within the return array 
    #so, iterating through the values 
    for ind in range(len(subgraphValues)): 
        returnCounts[int(subgraphValues[ind])] = subgraphCounts[ind]

    #pdb.set_trace()

    return returnValues, returnCounts

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
        self.subgraphList = []   
        self.index = index 

    def __str__(self):
        return "Node: "+(str(self.index))+" with neighbors: " +str(self.adjList)

class Edge(): 
    """
    Edge with slight addition 
    """

    def __init__(self, fromInd, toInd): 
        self.effectList = []
        self.fromInd = fromInd 
        self.toInd = toInd


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
    complementGraph = [Node(nodeInd) for nodeInd in range(len(graphIn))]  
    
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

    while(len(edgesToAddGraph1) != 0): 

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

def getBestierAdjList(numNodes, cliqueSize): 
    """
    Trying again. This method better jointly looks at an edge's effect on the previous graph. 
    
    Out of date documentation/procedure description 

    Something is wrong with this procedure though. It doesnt work with graph size 5, subgraph size 3
    
    1. Generate 2 arrays of edges to add 
    2. Get 2 sets of nodes     
    3. Get the dist. of subgraphs in each graph 
    4. Add an edge to the graph 
    5. Get and store the diff between new dist. of subgraphs and old dist. of subgraphs
    6. Remove the edge from the graph
    7. Repeat steps 4-6 for all edges, giving you the effect on subgraphs adding each edge has 
    8. Sort edges based on effect. So sort "# edges in full clique" times, in a stable fashion. 
    9. Pick first edge for graph 1, add it to graph 1, remove it from both possibilties
    10. Pick first edge for graph 2, add it to graph 2, remove it from both possibilities
    11. Repeat steps 3 - 10 until there are no more edges to add
    12. Return the new graph 
    """

    #1. Generate the edges we need to add 
    pairsToAddToGraph1 = get_index_combinations(numNodes, 2)
    pairsToAddToGraph2 = get_index_combinations(numNodes, 2)

    edges1 = [0]*len(pairsToAddToGraph1)
    edges2 = [0]*len(pairsToAddToGraph1)

    #init our edges 
    for i in range(len(edges1)): 
        #pdb.set_trace() 
        edges1[i] = Edge(pairsToAddToGraph1[i][0], pairsToAddToGraph1[i][1])
        edges2[i] = Edge(pairsToAddToGraph2[i][0], pairsToAddToGraph2[i][1])


    #2. Generate 2 lists of nodes, one for each graph  
    graph1 = [0]*numNodes
    graph2 = [0]*numNodes

    #initialize the nodes here 
    for i in range(numNodes): 
        graph1[i] = Node(i)
        graph2[i] = Node(i) 

    while(len(edges1) != 0): 
        
        print("These graphs")
        print("Graph 1")
        for i in range(len(graph1)): 
            print(graph1[i])
        print("Graph 2")
        for i in range(len(graph2)): 
            print(graph2[i])        

        #first, get the # subgraphs in each graph 
        initGraphCount1 = count_subgraphs(graph1, cliqueSize)
        initGraphCount2 = count_subgraphs(graph2, cliqueSize) 

        #then for each edge, get the diff/effect of adding it  
        for i in range(len(edges1)): 
            edgeToTryAdd1 = edges1[i] 
            
            #try adding it 
            graph1[edgeToTryAdd1.fromInd].adjList.add(edgeToTryAdd1.toInd)
            graph1[edgeToTryAdd1.toInd].adjList.add(edgeToTryAdd1.fromInd)

            #after you add it, recompute subgraph dist 
            subgraphDist = count_subgraphs(graph1, cliqueSize)

            #get diff between subgraph distributions 
            effect = subgraphDist[1] - initGraphCount1[1]

            edgeToTryAdd1.effectList = effect 

            #then remove the edge 
            graph1[edgeToTryAdd1.fromInd].adjList.remove(edgeToTryAdd1.toInd)
            graph1[edgeToTryAdd1.toInd].adjList.remove(edgeToTryAdd1.fromInd)


        #repeat for the other 
        for i in range(len(edges2)): 
            edgeToTryAdd2 = edges2[i]  

            #try adding it 
            graph2[edgeToTryAdd2.fromInd].adjList.add(edgeToTryAdd2.toInd)
            graph2[edgeToTryAdd2.toInd].adjList.add(edgeToTryAdd2.fromInd)

            #after you add it, recompute subgraph dist 
            subgraphDist = count_subgraphs(graph2, cliqueSize)

            #get diff  
            effect = subgraphDist[1] - initGraphCount2[1]
            edgeToTryAdd2.effectList = effect 

            #then remove the edge
            graph2[edgeToTryAdd2.fromInd].adjList.remove(edgeToTryAdd2.toInd)
            graph2[edgeToTryAdd2.toInd].adjList.remove(edgeToTryAdd2.fromInd)            

        #sort our edges based on this effect
        #if you know the limit of the entries in effect list, you could actually just 
        #compare amalagamated integers based on the entries  
        for sortInd in range(len(effect)):
            #sort each time by # equal to # of subgraphs with that many nodes 
            edges1 = sorted(edges1, key = lambda node: node.effectList[sortInd])
            edges2 = sorted(edges2, key = lambda node: node.effectList[sortInd])        
            
        #trying new approach. Just add the edge that has the min effect between the two graphs
        #this could actually create non-mirrored/balanced graphs. Maybe that will be good 
        #so, compare the two effect lists 
        choose1 = True 
        #for each entry in the effect list 
        for i in range(len(edges1[0].effectList)):  
            #compare them between effects list 
            if(edges1[0].effectList[i] > edges2[0].effectList[i]): 
                choose1 = False 
        
        if(choose1): 
            #after we sort these edges, pick the first remove from both...
            #repeat for other graph 
            edgeToAddGraph1 = edges1[0] 
            graph1[edgeToAddGraph1.fromInd].adjList.add(edgeToAddGraph1.toInd)
            graph1[edgeToAddGraph1.toInd].adjList.add(edgeToAddGraph1.fromInd)   
            
            toRemove = [0]
            #remove the matching edge from both 
            for edgeInd in range(len(edges1)): 
                    if(edges1[edgeInd].fromInd == edgeToAddGraph1.fromInd and edges1[edgeInd].toInd == edgeToAddGraph1.toInd):
                        toRemove = edges1[edgeInd]
            edges1.remove(toRemove)

            for edgeInd in range(len(edges2)):         
                if(edges2[edgeInd].fromInd == edgeToAddGraph1.fromInd and edges2[edgeInd].toInd == edgeToAddGraph1.toInd):
                    toRemove = edges2[edgeInd]
            edges2.remove(toRemove)

            # edges1.remove(edgeToAddGraph1)
            # edges2.remove(edgeToAddGraph1)

            #if we are empty, break 
            if(len(edges2) == 0): 
                break 
        else: 
            edgeToAddGraph2 = edges2[0] 
            graph2[edgeToAddGraph2.fromInd].adjList.add(edgeToAddGraph2.toInd)
            graph2[edgeToAddGraph2.toInd].adjList.add(edgeToAddGraph2.fromInd)     

            toRemove = [0]
            for edgeInd in range(len(edges1)): 
                if(edges1[edgeInd].fromInd == edgeToAddGraph2.fromInd and edges1[edgeInd].toInd == edgeToAddGraph2.toInd):
                    toRemove = edges1[edgeInd]
            edges1.remove(toRemove)
            
            for edgeInd in range(len(edges2)): 
                if(edges2[edgeInd].fromInd == edgeToAddGraph2.fromInd and edges2[edgeInd].toInd == edgeToAddGraph2.toInd): 
                    toRemove = edges2[edgeInd]     
            edges2.remove(toRemove)

    return graph1, graph2 

def getBestAdjList(numNodes, cliqueSize): 
    """
    Please note, this is a bad approach. It doesnt jointly consider nodes at all, so its really bad.
    
    What is this doing? This is implementing a new approach to generating an adjacency matrix of two binary graphs 
    that are respectively maximally spread out in connection space, as in, they each have a minimal number 
    of cliques based on edge selection.

    Please note, this "spread out" notion is with respect to a certain clique size.  

    What is the process? 
    1. Generate 2 arrays of edges to add 
    <Not>. Generate 2 adj list of sets (aghuarry dont do this, just make list of node objects)
    2. Get 2 sets of nodes 
    3. Get the dist. of subgraphs each node belongs to, for both sets of nodes 
    4. Sort both sets of nodes numEdgesToAdd - 1 times, using a stable sorting algorithm (could do it by numEdges, but wouldnt matter)
        1. First sort is by number of subgraphs each belong to that have one edge 
        2. Then sort by number of subgraphs each belong to with two edge s
        3 to numEdges-1. sort by number of subgraps each belong to with <step #> edges  
    5. Take the first two nodes of that sorting that dont have an edge...minimizing the upper one....
    6. Connect them, and remove that edge from both edges to add sets for each graph
    7. Repeat steps 6 & 7 for other graph 
    8. Repeat steps 4-8 until there are no edges to add  

    Input: 
    numNodes: number of nodes in our graph          
    cliqueSize: size clique we seek to minimize the # of 

    Output: 
    Good adjacency matrix for 2 complement graphs that sum to a fully connected graph 
    """
    #1. Generate the edges we need to add 
    pairsToAddToGraph1 = get_index_combinations(numNodes, 2)
    pairsToAddToGraph2 = get_index_combinations(numNodes, 2)

    #2. Generate 2 lists of nodes, one for each graph  
    nodeListGraph1 = [0]*numNodes
    nodeListGraph2 = [0]*numNodes

    #initialize the nodes here 
    for i in range(numNodes): 
        nodeListGraph1[i] = Node(i)
        nodeListGraph2[i] = Node(i) 

    #3. start loop here, so while we have edges to add in the first set
        #please note, that since we start adding with the first set, it will run out first
        #as in, with odd # edges, graph1 will have 1 more edge 
    while(len(pairsToAddToGraph1) != 0): 
        #sort each graph by index first for easier processing 
        nodeListGraph1 = sorted(nodeListGraph1, key = lambda node: node.index)
        nodeListGraph2 = sorted(nodeListGraph2, key = lambda node: node.index)

        #get the # of each subgraph each belong to
        indCounts1 = count_subgraphsSlightlyUpdated(nodeListGraph1, cliqueSize)
        indCounts2 = count_subgraphsSlightlyUpdated(nodeListGraph1, cliqueSize)

        #store counts 
        for nodeInd in range(numNodes): 
            nodeListGraph1[nodeInd].subgraphList = indCounts1[nodeInd] 
            nodeListGraph2[nodeInd].subgraphList = indCounts2[nodeInd] 

        #then, sort number of times equal to ind 
        for sortInd in range(np.shape(indCounts1)[1]):
            #sort each time by # equal to # of subgraphs with that many nodes 
            nodeListGraph1 = sorted(nodeListGraph1, key = lambda node: node.subgraphList[sortInd])
            nodeListGraph2 = sorted(nodeListGraph2, key = lambda node: node.subgraphList[sortInd])        

        #after sorting them, examine the first edge we can add for each 
        #this will be the the first two nodes in nodeListGraph1 that appear as a pair
        #in the pair array 
        #first two meaning, we minimize the index of the second one. 
        #so [0,1] then [0,2] then [1,2]
        
        #so, get the edge to add for the first graph 
        edgeToAdd = []
        found1 = False 
        #iterating through our many-sorted node list  
        for nodeFromInd in range(numNodes): 
            for nodeToInd in range(nodeFromInd): 
                #if this edge exists: 
                #pdb.set_trace() 
                if((nodeToInd, nodeFromInd) in pairsToAddToGraph1): 
                    edgeToAdd = [nodeToInd, nodeFromInd] 
                    found1 = True 
                    break 
            if(found1): 
                break 
        
        #add the edge to the first graph, remove it from both possibilities
        #so, sort and then index by the corresponding node #  
        nodeListGraph1 = sorted(nodeListGraph1, key = lambda node: node.index)
        #print("Hello")
        #pdb.set_trace() 
        nodeListGraph1[edgeToAdd[0]].adjList.add(edgeToAdd[1]) 
        nodeListGraph1[edgeToAdd[1]].adjList.add(edgeToAdd[0]) 
        
        #remove the pair from each set 
        pairsToAddToGraph1.remove((nodeToInd,nodeFromInd)) 
        pairsToAddToGraph2.remove((nodeToInd,nodeFromInd)) 

        #in odd # pairs, at final iteration both will be empty and can exit 
        if(len(pairsToAddToGraph1) == 0): 
            break 
        #in even # pairs, at final iteration, both will have one more, and we can then add it here 
        #could functionalize this...later 

        #so, get the edge to add for the second graph 
        edgeToAdd2 = []
        found2 = False 
        #iterating through our many-sorted node list  
        for nodeFromInd in range(numNodes): 
            for nodeToInd in range(nodeFromInd): 
                #if this edge exists: 
                if((nodeToInd, nodeFromInd) in pairsToAddToGraph2): 
                    edgeToAdd2 = [nodeToInd, nodeFromInd] 
                    found2 = True 
                    break 
            if(found2): 
                break 
        
        #add the edge to the first graph, remove it from both possibilities
        #so, sort and then index by the corresponding node #  
        nodeListGraph2 = sorted(nodeListGraph2, key = lambda node: node.index)
        nodeListGraph2[edgeToAdd2[0]].adjList.add(edgeToAdd2[1]) 
        nodeListGraph2[edgeToAdd2[1]].adjList.add(edgeToAdd2[0]) 
        
        #get num directional edges...
        sum = 0 
        for node in nodeListGraph2: 
            sum+=len(node.adjList)

        #remove the pair from each set 
        pairsToAddToGraph1.remove((nodeToInd, nodeFromInd)) 
        pairsToAddToGraph2.remove((nodeToInd, nodeFromInd)) 

    #return the data 
    return nodeListGraph1, nodeListGraph2

#for node in graph: 
#    print(node)

#exit() 
#graph1, graph2 = getBestAdjList(6,3) 
#print(graph1)


#what are we doing? Testing basic cases of ramsey numbers with this. 
# numNodes = 18
# numSizeSubgraph = 4
# numEdgesInClique = numSizeSubgraph*(numSizeSubgraph-1)/2 
# numEdgesToAdd = int(np.ceil(numNodes*(numNodes-1)/4)) 

# adjList1, adjList2 = getComplementBalancedGoodAdjList(numNodes)

# print("Num edges in each subgraph of size 4")
# print(count_subgraphs(adjList1, numSizeSubgraph))
# print(count_subgraphs(adjList2, numSizeSubgraph))


# adjList = getGoodAdjListModified(numNodes, numEdgesToAdd)
# compAdjList = getGraphComplement(adjList) 

# print("Num edges in each subgraph of size 4")
# print(count_subgraphs(adjList,numSizeSubgraph))
# print(count_subgraphs(compAdjList, numSizeSubgraph))


# numEdgesRequired = int(np.ceil(numNodes*(numNodes-1)/4))
# goodAdjList = getGoodAdjListModified(numNodes, numEdgesRequired)
# comp = getGraphComplement(goodAdjList)

# g1, g2 = getComplementBalancedGoodAdjList(numNodes)

# result1 = count_subgraphs(g1, numSizeSubgraph)
# print(result1) 

# result2 = count_subgraphs(g2, numSizeSubgraph)
# print(result2) 

# print("Number Edges in Full Clique")
# print(numEdgesInClique)







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


