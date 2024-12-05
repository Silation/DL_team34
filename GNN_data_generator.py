from concorde.tsp import TSPSolver
from random_node_generator import *
import os
import random
import torch
from torch_geometric.data import Data
import math
import shutil

def read_tsp_file(file_path):
    """
    TSPLIB 형식의 .tsp 파일을 읽어 좌표 데이터를 반환합니다.
    """
    coordinates = []
    start_reading = False
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == "EOF":
                break
            
            if start_reading:
                parts = line.strip().split()
                if len(parts) == 3:
                    _, x, y = parts
                    coordinates.append((float(x), float(y)))

            if line.strip() == "NODE_COORD_SECTION":
                start_reading = True

    return coordinates


def make_test_dataset(data_amount=100, folder_path='pt_larger_dataset'):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i in range(2551, data_amount + 2551):
        #num_nodes = random.randint(50, 200)
        num_nodes = random.randint(50, 1000)
        coordinates = generate_random_nodes(num_nodes)
        solver = TSPSolver.from_data(
            [x for x, y in coordinates], 
            [y for x, y in coordinates], 
            norm="EUC_2D"
        )

        try:
            solution = solver.solve()
        except Exception as e:
            print(f"Error solving TSP for sample {i}: {e}")
            continue

        optimal_edges = set()
        for node in range(len(solution.tour) - 1):
            from_node, to_node = solution.tour[node], solution.tour[node + 1]
            optimal_edges.add((from_node, to_node))

        # Debugging optimal edges
        #print(f"Optimal edges for sample {i}: {optimal_edges}")

        x = torch.tensor(coordinates, dtype=torch.float)
        edge_index = []
        edge_attr = []

        for from_node in range(len(coordinates)):
            for to_node in range(len(coordinates)):
                if from_node != to_node:
                    edge_index.append([from_node, to_node])
                    distance = math.sqrt(
                        (coordinates[from_node][0] - coordinates[to_node][0])**2 +
                        (coordinates[from_node][1] - coordinates[to_node][1])**2
                    )
                    is_in_tour = 1 if (from_node, to_node) in optimal_edges else 0
                    edge_attr.append([is_in_tour, distance])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Debugging edge attributes
        num_in_tour = (edge_attr[:, 0] == 1).sum().item()
        print(f"Number of edges in optimal tour for sample {i}: {num_in_tour}")
        #print(f"Edge Labels (is_in_tour) for sample {i}: {edge_attr[:, 0]}")

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        torch.save(graph_data, f'{folder_path}/tsp_graph_data_{i}.pt')
    
def process_tsp_files(tsp_folder_path, output_folder_path):
    """
    주어진 폴더 내 모든 .tsp 파일을 읽어 .pt 형식으로 변환합니다.
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for tsp_file in os.listdir(tsp_folder_path):
        if tsp_file.endswith(".tsp"):
            file_path = os.path.join(tsp_folder_path, tsp_file)
            coordinates = read_tsp_file(file_path)

            solver = TSPSolver.from_data(
                [x for x, y in coordinates],
                [y for x, y in coordinates],
                norm="EUC_2D"
            )

            try:
                solution = solver.solve()
            except Exception as e:
                print(f"Error solving TSP for {tsp_file}: {e}")
                continue

            optimal_edges = set()
            for node in range(len(solution.tour) - 1):
                from_node, to_node = solution.tour[node], solution.tour[node + 1]
                optimal_edges.add((from_node, to_node))

            x = torch.tensor(coordinates, dtype=torch.float)
            edge_index = []
            edge_attr = []

            for from_node in range(len(coordinates)):
                for to_node in range(len(coordinates)):
                    if from_node != to_node:
                        edge_index.append([from_node, to_node])
                        distance = math.sqrt(
                            (coordinates[from_node][0] - coordinates[to_node][0]) ** 2 +
                            (coordinates[from_node][1] - coordinates[to_node][1]) ** 2
                        )
                        is_in_tour = 1 if (from_node, to_node) in optimal_edges else 0
                        edge_attr.append([is_in_tour, distance])

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            output_file = os.path.join(output_folder_path, f"{os.path.splitext(tsp_file)[0]}.pt")
            torch.save(graph_data, output_file)
            print(f"Saved {output_file}")



# 사용 예제
tsp_folder_path = "DL_TestCase"  # .tsp 파일이 있는 폴더 경로
output_folder_path = "pt_new_dataset_test"       # 변환된 .pt 파일을 저장할 폴더 경로
process_tsp_files(tsp_folder_path, output_folder_path)

#make_test_dataset(data_amount=9000)