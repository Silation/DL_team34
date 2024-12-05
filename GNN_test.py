import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.loader import DataLoader
from GNN_train_2 import TSPGNN
import os
import shutil
import pandas as pd


# 모델 정의 및 가중치 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TSPGNN().to(device)

# 학습된 가중치 파일 로드
try:
    #model.load_state_dict(torch.load('tsp_gnn_edge_model.pth', map_location=device))
    model.load_state_dict(torch.load('larger_tsp_gnn_edge_model.pth', map_location=device))
    
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

model.eval()

# 테스트 데이터셋 정의
class TSPTestDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path='pt_new_dataset_test'):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in the specified folder: {folder_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        graph_data = torch.load(self.files[idx])
        return graph_data

# 데이터 로더 설정
test_dataset = TSPTestDataset('pt_new_dataset_test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# 네트워크 정보 저장 함수
def save_network_data_per_test(data, edge_pred_sigmoid, test_id, save_dir="network_info"):
    """
    테스트 파일별로 네트워크 정보를 저장합니다.
    """
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성

    edges = data.edge_index.cpu().t().numpy()  # 엣지 정보 (from_node, to_node)
    nodes = data.x.cpu().numpy()  # 노드 좌표 (x, y)
    predicted_probs = edge_pred_sigmoid.cpu().numpy()  # 예측 확률
    
    # 노드 데이터프레임 생성
    node_df = pd.DataFrame(nodes, columns=["x", "y"])
    node_df["node_id"] = range(len(nodes))
    
    # 엣지 데이터프레임 생성
    edge_data = []
    for i, (from_node, to_node) in enumerate(edges):
        distance = ((nodes[from_node][0] - nodes[to_node][0])**2 + 
                    (nodes[from_node][1] - nodes[to_node][1])**2)**0.5
        edge_data.append({
            "from_node": from_node,
            "to_node": to_node,
            "distance": distance,
            "predicted_probability": predicted_probs[i],
            "is_in_tour": 1 if predicted_probs[i] >= 0.5 else 0
        })
    
    edge_df = pd.DataFrame(edge_data)
    
    # 파일 저장
    node_file_path = os.path.join(save_dir, f"{test_id}_nodes.csv")
    edge_file_path = os.path.join(save_dir, f"{test_id}_edges.csv")
    node_df.to_csv(node_file_path, index=False)
    edge_df.to_csv(edge_file_path, index=False)
    
    print(f"Test {test_id} - Nodes saved to {node_file_path}")
    print(f"Test {test_id} - Edges saved to {edge_file_path}")
    

# 그래프 시각화 함수
def visualize_tsp(data, edge_pred_sigmoid, binary_preds, edge_labels, title="TSP Graph", save_path_pred=None, save_path_gt=None):
    """
    두 개의 그래프를 각각 저장:
    1. 예측된 엣지 강조 (빨간색)
    2. 정답 엣지 강조 (파란색)
    """
    pos = {i: (x.item(), y.item()) for i, (x, y) in enumerate(data.x.cpu())}
    edges = data.edge_index.cpu().t().tolist()

    # 그래프 1: 예측된 엣지 강조
    graph_pred = nx.Graph()
    for i, (u, v) in enumerate(edges):
        graph_pred.add_edge(u, v, weight=edge_pred_sigmoid[i].item())
    
    plt.figure(figsize=(10, 10))
    nx.draw(
        graph_pred,
        pos,
        with_labels=True,
        node_size=300,
        font_size=8,
        edge_color="gray",
        alpha=0.5,
        width=0.1
    )
    # 예측 엣지 강조 (빨간색)
    for i, (u, v) in enumerate(edges):
        if binary_preds[i].item() == 1.0:
            nx.draw_networkx_edges(
                graph_pred, pos, edgelist=[(u, v)], edge_color="red", width=2.0
            )
    plt.title(f"{title} - Predictions")
    if save_path_pred:
        plt.savefig(save_path_pred)
        print(f"Prediction graph saved at {save_path_pred}")
    plt.close()

    # 그래프 2: 정답 엣지 강조
    graph_gt = nx.Graph()
    for i, (u, v) in enumerate(edges):
        graph_gt.add_edge(u, v, weight=edge_pred_sigmoid[i].item())
    
    plt.figure(figsize=(10, 10))
    nx.draw(
        graph_gt,
        pos,
        with_labels=True,
        node_size=300,
        font_size=8,
        edge_color="gray",
        alpha=0.5,
        width=0.1
    )
    # 정답 엣지 강조 (파란색)
    for i, (u, v) in enumerate(edges):
        if edge_labels[i].item() == 1.0:
            nx.draw_networkx_edges(
                graph_gt, pos, edgelist=[(u, v)], edge_color="blue", width=2.0
            )
    plt.title(f"{title} - Ground Truth")
    if save_path_gt:
        plt.savefig(save_path_gt)
        print(f"Ground truth graph saved at {save_path_gt}")
    plt.close()


# 모델 평가 함수
def evaluate_model_and_visualize(model, test_loader):
    """
    모델 평가 및 시각화 함수
    """
    model.eval()
    total_correct = 0
    total_positive = 0
    total_predicted_positive = 0
    total_logistic_loss = 0.0
    total_edges = 0

    bce_loss = torch.nn.BCELoss()  # Logistic loss 계산
    
    # 시각화 및 엣지 예측 디렉토리 설정
    save_dir = "GNN_visualized_graphs"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    edge_prob_dir = "edge_predictions"
    if os.path.exists(edge_prob_dir):
        shutil.rmtree(edge_prob_dir)
    os.makedirs(edge_prob_dir)
    
    # 네트워크 데이터 저장 디렉토리 추가
    network_info_dir = "network_info"
    if not os.path.exists(network_info_dir):
        os.makedirs(network_info_dir)

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            
            all_edges = []  # 모든 엣지 정보를 저장할 리스트
            data = data.to(device)

            # 모델 예측
            edge_pred = model(data.x, data.edge_index, data.edge_attr)

            # Sigmoid를 통해 예측 확률로 변환
            edge_pred_sigmoid = torch.sigmoid(edge_pred)

            # 정답 라벨
            edge_labels = data.edge_attr[:, 0]  # 최적 경로 포함 여부

            # Binary Prediction (0.5 임계값)
            binary_preds = (edge_pred_sigmoid >= 0.5).float()
            binary_preds = binary_preds.to(edge_labels.device)  # 텐서로 변환 및 디바이스 일치

            # 엣지별 정보 저장
            edge_index = data.edge_index.cpu().t().numpy()  # 엣지 연결 정보 (from_node, to_node)
            predicted_probs = edge_pred_sigmoid.cpu().numpy()  # 예측 확률
            binary_preds_np = binary_preds.cpu().numpy()  # 이진 예측
            ground_truth = edge_labels.cpu().numpy()  # 실제 레이블

            # 데이터프레임으로 저장할 리스트 생성
            for i, (from_node, to_node) in enumerate(edge_index):
                all_edges.append({
                    "from_node": from_node,
                    "to_node": to_node,
                    "predicted_probability": predicted_probs[i],
                    "binary_prediction": binary_preds_np[i],
                    "ground_truth": ground_truth[i]
                })

            # Precision, Recall 계산용
            total_correct += (binary_preds * edge_labels).sum().item()  # True Positive
            total_positive += edge_labels.sum().item()  # Ground Truth Positive
            total_predicted_positive += binary_preds.sum().item()  # Predicted Positive

            # Logistic Loss 계산
            total_logistic_loss += bce_loss(edge_pred_sigmoid, edge_labels).item() * edge_labels.size(0)

            # 전체 엣지 수
            total_edges += edge_labels.size(0)

            # 시각화
            if batch_idx < 10:  # 처음 10개 배치만 시각화
                save_path_pred = os.path.join(save_dir, f"tsp_graph_{batch_idx + 1}_pred.png")
                save_path_gt = os.path.join(save_dir, f"tsp_graph_{batch_idx + 1}_gt.png")
                visualize_tsp(data, edge_pred_sigmoid, binary_preds, edge_labels, 
                              title=f"TSP Graph {batch_idx + 1}", 
                              save_path_pred=save_path_pred, 
                              save_path_gt=save_path_gt)

            # 데이터프레임 생성 및 저장 (엣지 예측 결과)
            df = pd.DataFrame(all_edges)
            df.to_csv(f"{edge_prob_dir}/edge_predictions_{batch_idx+1}.csv", index=False)
            print(f"Edge predictions saved to edge_predictions_{batch_idx+1}.csv")

            # 추가: 네트워크 정보 저장
            network_file_path = os.path.join(network_info_dir, f"test_{batch_idx + 1}_network.csv")
            save_network_data(data, edge_pred_sigmoid, file_path=network_file_path)

    # Precision, Recall 계산
    precision = total_correct / total_predicted_positive if total_predicted_positive > 0 else 0.0
    recall = total_correct / total_positive if total_positive > 0 else 0.0

    # Average Logistic Loss
    avg_logistic_loss = total_logistic_loss / total_edges if total_edges > 0 else 0.0

    print(f"Optimal Edges Precision: {precision * 100:.5f}%")
    print(f"Optimal Edges Recall: {recall * 100:.5f}%")
    print(f"Average Logistic Loss: {avg_logistic_loss:.10f}")

# 네트워크 정보 저장 함수
def save_network_data(data, edge_pred_sigmoid, file_path="network_info.csv"):
    """
    네트워크 정보를 CSV로 저장
    """
    edges = data.edge_index.cpu().t().numpy()  # 엣지 정보 (from_node, to_node)
    nodes = data.x.cpu().numpy()  # 노드 좌표 (x, y)
    predicted_probs = edge_pred_sigmoid.cpu().numpy()  # 예측 확률
    
    # 노드 데이터프레임
    node_df = pd.DataFrame(nodes, columns=["x", "y"])
    node_df["node_id"] = range(len(nodes))
    
    # 엣지 데이터프레임
    edge_data = []
    for i, (from_node, to_node) in enumerate(edges):
        distance = ((nodes[from_node][0] - nodes[to_node][0])**2 + 
                    (nodes[from_node][1] - nodes[to_node][1])**2)**0.5
        edge_data.append({
            "from_node": from_node,
            "to_node": to_node,
            "distance": distance,
            "predicted_probability": predicted_probs[i],
            "is_in_tour": 1 if predicted_probs[i] >= 0.5 else 0
        })
    
    edge_df = pd.DataFrame(edge_data)
    
    # 저장
    node_df.to_csv(file_path.replace(".csv", "_nodes.csv"), index=False)
    edge_df.to_csv(file_path.replace(".csv", "_edges.csv"), index=False)
    print(f"Network data saved to {file_path.replace('.csv', '_nodes.csv')} and {file_path.replace('.csv', '_edges.csv')}")

# 모델 평가 및 시각화 실행
evaluate_model_and_visualize(model, test_loader)
