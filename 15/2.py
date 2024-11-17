from math import sqrt
import subprocess

class Point:
    def __init__(self, P):
        self.x = P[0]
        self.y = P[1]
    def dist(self, p) -> float:
        return sqrt((p.x - self.x)**2 + (p.y - self.y)**2)

class DSU:
    """
    Class for Disjoint Set Union implementaion
    """
    def __init__(self, n):
        self.n = n
        self.par = [i for i in range(n)]
    
    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    
    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        if x == y:
            return False
        self.par[x] = y
        return True

class TSP:
    def __init__(self, points):
        self.n = len(points)
        self.points = points
        self.MST = [[0 for _ in range(self.n)] for _ in range(self.n)]
        self.sequence = []
        self.TSP = []

        self.generate_MST()
        self.MCPM()
        self.euler_circuit_dfs(0)
        self.generate_TSP()

    def generate_MST(self):
        """
        generate MST graph of points
        """
        edges = []
        UFHandler = DSU(self.n)
        for idx1 in range(self.n):
            for idx2 in range(idx1 + 1, self.n):
                edges.append((idx1, idx2))
        edges.sort(key = lambda x: self.points[x[0]].dist(self.points[x[1]]))
        for idx1, idx2 in edges:
            if UFHandler.union(idx1, idx2) == True:
                self.MST[idx1][idx2] = 1
                self.MST[idx2][idx1] = 1

    def MCPM(self):
        """
        function for minimum cost perfect matching.
        https://github.com/dilsonpereira/Minimum-Cost-Perfect-Matching
        There are only c++ implementation to handle MCPM, so I used subprocess to use it as python.
        """
        odd_node = []
        for idx in range(self.n):
            if sum(self.MST[idx]) % 2 == 1:
                odd_node.append(idx)
        odd_size = len(odd_node)

        file_input = open('input.txt', 'w')
        file_input.write(str(odd_size) + '\n')
        file_input.write(str(odd_size*(odd_size - 1)//2) + '\n')
        for i1 in range(odd_size):
            for i2 in range(i1 + 1, odd_size):
                idx1, idx2 = odd_node[i1], odd_node[i2]
                dist = self.points[idx1].dist(self.points[idx2])
                file_input.write(str(i1) + ' ' + str(i2) + ' ' + str(dist) + '\n')
        file_input.close()

        c_executable_path = './Minimum-Cost-Perfect-Matching/mcpm'
        args = ['-f', './input.txt', '--minweight']

        process = subprocess.Popen(
            [c_executable_path] + args,
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True
        )

        output, error = process.communicate(input='')
        output = output.split('\n')[2:-1]
        
        # add matched edges to MST
        for edges in output:
            i1, i2 = map(int, edges.split())
            idx1, idx2 = odd_node[i1], odd_node[i2]
            self.MST[idx1][idx2] = 1
            self.MST[idx2][idx1] = 1

    def euler_circuit_dfs(self, v):
        """
        generate Euler Circuit with Hierholzer's Algorithm
        """
        path = []
        stack = [v]
        
        while stack:
            u = stack[-1]
            has_edge = False
            for w in range(self.n):
                if self.MST[u][w] == 1:
                    stack.append(w)
                    self.MST[u][w] = self.MST[w][u] = 0
                    has_edge = True
                    break
            if not has_edge:
                path.append(stack.pop())
        
        self.sequence = path[::-1]

    def DFS(self, node, par):
        """
        generate DFS visit-sequence of MST

            node : current node
            par : parent node of node
        """
        self.sequence.append(node)
        for next_node in range(self.n):
            if self.MST[node][next_node] and par != next_node:
                self.DFS(next_node, node)
                self.sequence.append(node)

    def generate_TSP(self):
        """
        generate Traversal Salesman Path with 2-approximation through DFS visit-sequence.
        """

        visited = [0 for _ in range(self.n)]
        for idx in self.sequence:
            if not visited[idx]:
                self.TSP.append(idx)
                visited[idx] = 1

# input format based on mingi code
points = {0: (42, 94), 1: (89, 78), 2: (36, 15), 3: (30, 54), 4: (86, 95), 5: (66, 74), 6: (20, 50), 7: (11, 7), 8: (63, 63), 9: (66, 68), 10: (95, 44), 11: (15, 64), 12: (95, 55), 13: (9, 30), 14: (29, 9), 15: (27, 94), 16: (93, 32), 17: (23, 32), 18: (61, 83), 19: (22, 99), 20: (96, 98), 21: (10, 71), 22: (16, 10), 23: (99, 71), 24: (91, 54), 25: (42, 86), 26: (70, 76), 27: (71, 21), 28: (35, 23), 29: (88, 40)}

for i in points:
    points[i] = Point(points[i])

u = TSP(points)
print(u.TSP)