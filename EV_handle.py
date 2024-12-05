class EV_Handler():
    def __init__(self, nodes, EV):
        self.n = len(nodes)
        self.nodes = nodes
        self.EV = EV
    
    def ev_greedy(self):
        """
        Implements the EV-greedy algorithm.
        
        Parameters:
            EV (numpy.ndarray): n x n array of Good-Edge Values.
            
        Returns:
            list: A list of edges (i, j) that form the tour.
        """
        n = self.n  # Number of nodes
        Graph = [['E' for _ in range(n)] for _ in range(n)] 
        E = [(i, j) for i in range(n) for j in range(i+1, n)]
        Z = []
        def makes_tour_close(edge):
            """
            Checks if adding the edge to the current solution makes a cycle prematurely.
            """
            i, j = edge
            assert Graph[i][j] == Graph[j][i] == 'E'
            Graph[i][j] = 'Z'
            Graph[j][i] = 'Z'
            
            visited = {}
            # Simple cycle detection using DFS
            def dfs(node, parent):
                visited[node] = True
                for next_node in range(self.n):
                    if next_node == node or Graph[node][next_node] == 'E':
                        continue
                    if next_node not in visited:
                        if dfs(next_node, node):
                            return True
                    elif next_node != parent:
                        return True
                return False
            
            if dfs(i, -1):
                Graph[i][j] = 'E'
                Graph[j][i] = 'E'
                return True
            else:
                return False

        while len(Z) < n - 1:
            # Find the edge (i, j) with the maximum Good-Edge Value in E_prime
            i, j = max(E, key=lambda edge: self.EV[edge[0]][edge[1]])
            E.remove((i, j))  # Remove (i, j) from E_prime

            # Count the number of edges connected to each node in Z
            connected_edges = [0 for _ in range(n)]
            for u, v in Z:
                connected_edges[u] += 1
                connected_edges[v] += 1
            
            if connected_edges[i] < 2 and connected_edges[j] < 2 and not makes_tour_close((i, j)):
                Z.append((i, j))  # Add (i, j) to Z
        
        connected_edges = [0 for _ in range(n)]
        for u, v in Z:
            connected_edges[u] += 1
            connected_edges[v] += 1
        
        start = 0
        while connected_edges[start] == 2:
            start += 1
        
        path = [start]
        for i in range(n-1):
            for u, v in Z:
                if u == start:
                    start = v
                    Z.remove((u, v))
                    break
                if v == start:
                    start = u
                    Z.remove((u, v))
                    break
            path.append(start)
        
        assert len(Z) == 0
        return path

    def ev_2opt(self, initial_tour):
        """
        Implements the EV-2opt algorithm.
        
        Parameters:
            EV (numpy.ndarray): n x n array of Good-Edge Values.
            initial_tour (list): Initial tour as a list of node indices.
            
        Returns:
            list: Optimized tour.
        """
        n = self.n  # Number of nodes in the tour
        tour = initial_tour[:]  # Make a copy of the initial tour for modification

        while True:
            improved = False

            # Try to find a pair of edges to swap that improves the tour
            for i in range(n - 1):
                for j in range(i + 2, n):  # Ensure non-adjacent edges
                    if j == n - 1 and i == 0:  # Skip if swapping would break the cycle
                        continue
                    # Current edges in the tour
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[(j + 1) % n]
                    
                    # Check if swapping improves the tour
                    if self.EV[a][b] + self.EV[c][d] < self.EV[a][c] + self.EV[b][d]:
                        # Perform the 2-opt swap: reverse the segment between i+1 and j
                        tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                        improved = True
                        break  # Break inner loop
                if improved:
                    break  # Break outer loop
            
            if not improved:
                break  # Stop if no improvement is found

        return tour
    
"""
    Usage: 

    # node := matrices with n points
    # example) [(0,0), (1,3), (5,1), ...]
    # EV := n x n matrices, which EV[i][j] means the value of edge (i, j). Recommand float value.
    
    U = EV_Handler(node, EV)
    path_greedy = U.ev_greedy()  # returns path [n1, n2, n3, ...]

    initial_tour = [i for i in range(n)]
    # ok to set initial_tour = path_greedy

    path_2opt = U.ev_2opt(initial_tour) # returns path [n1, n2, n3, ...]


"""