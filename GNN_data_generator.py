from concorde.tsp import TSPSolver
from random_node_generator import *
import pandas as pd
import os
import random
import torch
from torch_geometric.data import Data

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
    
    if not os.path.exists('pt_dataset'): os.makedirs('pt_dataset')
    
    for i in range(501, data_amount+501):
        # 랜덤 좌표 생성 및 TSP 솔버 초기화
        coordinates = generate_random_nodes(random.randint(50, 200))
        solver = TSPSolver.from_data([x for x, y in coordinates], [y for x, y in coordinates], norm="EUC_2D")
        
        # 최적 경로 계산
        solution = solver.solve()
        
        # 최적 경로 엣지 정보 저장
        optimal_edges = set()
        for node in range(len(solution.tour) - 1):
            from_node, to_node = solution.tour[node], solution.tour[node + 1]
            optimal_edges.add((from_node, to_node))
        
        # 노드 좌표를 텐서로 변환
        x = torch.tensor(coordinates, dtype=torch.float)  # 노드 특징 (좌표 정보)

        # 엣지 정보 생성
        edge_index = []
        edge_attr = []
        
        for from_node in range(len(coordinates)):
            for to_node in range(len(coordinates)):
                if from_node != to_node:
                    # 엣지 연결 정보
                    edge_index.append([from_node, to_node])
                    # 최적 경로 포함 여부
                    is_in_tour = 1 if (from_node, to_node) in optimal_edges else 0
                    edge_attr.append([is_in_tour])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # (2, num_edges) 형식
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # (num_edges, 1) 형식
        
        # 그래프 데이터 생성
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # 그래프 데이터를 .pt 파일로 저장
        torch.save(graph_data, f'pt_dataset/tsp_graph_data_{i}.pt')


'''
# 테스트 폴더 경로
folder_path = 'DL_TestCase/'
coordinates = read_tsp_file(folder_path + 'kroA200.tsp')
'''

make_dataset(4500)