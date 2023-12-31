def count_subgraphs(graph):
    n = len(graph)
    max_edges = max(map(len, graph))
    
    # Initialize a 2D table to store subgraph counts
    dp = [[0] * (max_edges + 1) for _ in range(n + 1)]
    
    # Base case: there is one way to form a subgraph with 0 edges
    dp[0][0] = 1
    
    # Iterate through edges and possible edge counts
    for i in range(1, n + 1):
        for j in range(max_edges + 1):
            # Exclude the i-th edge
            dp[i][j] = dp[i - 1][j]
            
            # Include the i-th edge if it doesn't exceed the current edge count
            if j >= len(graph[i - 1]):
                dp[i][j] += dp[i - 1][j - len(graph[i - 1])]
    
    # The final result is in dp[n][j], where n is the number of edges
    return dp[n]

def getGoodAdjacencyList(numNodes, numEdgesToAdd): 
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

        #then, iterate through and find the first two nodes that dont have each other 
        #as neighbors
        #please note, we are prioritizing minimizing the index of the max one 

        for node in nodeList:
            print(node)
        
        pdb.set_trace()

        madeConnection = False 
        for node2Ind in range(numNodes):
            
            #once we have the list of sorted nodes, then i need to take the list 
            #of nodes that up until we have a 3rd unique length of adj list
            subList = [] 
            
            #iterate through our entire node list (at least try to) 
            adjListSizes = set() 
            for subListNodeInd in range(numNodes):
                #add the new length to it 
                adjListSizes.add(len(nodeList[subListNodeInd].adjList))
                #if our size is bigger than required, then stop 
                if(len(adjListSizes) > 2): 
                    break  
                subList = subList + [nodeList[subListNodeInd]] 
            

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


        #create storage for sub list of nodes 
        subList = [] 
        
        #create storage for unique sizes 
        adjListSizes = set() 

        #iterate through our entire node list (at least try to) 
        for subListNodeInd in range(numNodes):
            #add the new length to it 
            adjListSizes.add(len(nodeList[subListNodeInd].adjList))
            #if our size is bigger than required, then stop 
            if(len(adjListSizes) > 2): 
                break  
            subList = subList + [nodeList[subListNodeInd]] 


adjacency_lists = [
    [1, 2, 3, 4,5],  # Node 0
    [0],              # Node 1
    [0],              # Node 2
    [0],              # Node 3
    [0],              # Node 4
    [0]               # Node 5
]

graph = create_nodes(adjacency_lists)
stuff = count_subgraphs(graph,3)
print(stuff)

            