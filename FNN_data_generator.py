from concorde.tsp import TSPSolver
from random_node_generator import *
import pandas as pd
import os
import random

def read_tsp_file(file_path):
    # coordinates에 (x, y) tuple 저장
    coordinates = []
    start_reading = False
    
    with open(file_path, 'r') as file:
        for line in file:
            
            if line.strip() == "EOF":
                break
            
            if start_reading:
                _, x, y = line.split()
                coordinates.append((int(x), int(y)))

            if line.strip() == "NODE_COORD_SECTION":
                start_reading = True

    return coordinates

def make_dataset(data_amount = 100):
    
    if not os.path.exists('dataset'): os.makedirs('dataset')
    
    for i in range(501, data_amount+501):
        # nodes list를 생성합니다 [(x,y)]
        # node 갯수: [50, 200]
        coordinates = generate_random_nodes(random.randint(50, 200))
        #print(coordinates)

        # TSPSolver를 사용하여 문제 인스턴스를 생성합니다.
        solver = TSPSolver.from_data([x for x, y in coordinates], [y for x, y in coordinates], norm="EUC_2D")

        # 최적화를 실행합니다.
        solution = solver.solve()
        
        # 최적 경로 엣지 정보 저장
        optimal_edges = set()
        for node in range(len(solution.tour) - 1):
            from_node, to_node = solution.tour[node], solution.tour[node + 1]
            from_x, from_y = coordinates[from_node]
            to_x, to_y = coordinates[to_node]
            optimal_edges.add((from_node, to_node))
        
        # 모든 노드 간 엣지를 포함하여 DataFrame 생성
        edge_data = []
        for from_node in range(len(coordinates)):
            for to_node in range(len(coordinates)):
                if from_node != to_node:
                    from_x, from_y = coordinates[from_node]
                    to_x, to_y = coordinates[to_node]
                    # 최적 경로에 포함 여부 확인
                    is_in_tour = 1 if (from_node, to_node) in optimal_edges else 0
                    edge_data.append((from_node, to_node, from_x, from_y, to_x, to_y, is_in_tour))
        

        # DataFrame 생성 및 CSV 저장
        tsp_data = pd.DataFrame(edge_data, columns=['from_node', 'to_node', 'from_x', 'from_y', 'to_x', 'to_y', 'is_in_tour'])
        tsp_data.to_csv(f'dataset/tsp_data_{i}.csv', index=False)


'''
# 테스트 폴더 경로
folder_path = 'DL_TestCase/'
coordinates = read_tsp_file(folder_path + 'kroA200.tsp')
'''


# 데이터셋 생성
make_dataset(7500)


