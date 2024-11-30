import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.loader import DataLoader
from GNN_train_2 import TSPGNN
import os
import shutil

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
    def __init__(self, folder_path='pt_dataset_test'):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in the specified folder: {folder_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        graph_data = torch.load(self.files[idx])
        return graph_data

# 데이터 로더 설정
test_dataset = TSPTestDataset('pt_dataset_test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 그래프 시각화 함수
def visualize_tsp(data, edge_pred_sigmoid, binary_preds, edge_labels, title="TSP Graph", save_path=None):
    """
    시각화를 위해 NetworkX 그래프를 생성 및 시각화
    """
    graph = nx.Graph()
    pos = {i: (x.item(), y.item()) for i, (x, y) in enumerate(data.x.cpu())}

    # 모든 엣지 추가 (회색)
    edges = data.edge_index.cpu().t().tolist()
    for i, (u, v) in enumerate(edges):
        graph.add_edge(u, v, weight=edge_pred_sigmoid[i].item())
    
    plt.figure(figsize=(10, 10))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=300,
        font_size=8,
        edge_color="gray",
        alpha=0.5,
        width=0.1
    )

    # 정답 엣지(1) 강조 (파란색)
    for i, (u, v) in enumerate(edges):
        if edge_labels[i].item() == 1.0:
            nx.draw_networkx_edges(
                graph, pos, edgelist=[(u, v)], edge_color="blue", width=2.0
            )

    # 예측 엣지(0.5 이상) 강조 (빨간색)
    for i, (u, v) in enumerate(edges):
        if binary_preds[i].item() == 1.0:
            nx.draw_networkx_edges(
                graph, pos, edgelist=[(u, v)], edge_color="red", width=1.5
            )

    plt.title(title)
    plt.show()
    if save_path:
        plt.savefig(save_path)  # 그래프 저장
        print(f"Graph saved at {save_path}")
    plt.close()

# 모델 평가 함수
def evaluate_model_and_visualize(model, test_loader):
    model.eval()
    total_correct = 0
    total_positive = 0
    total_predicted_positive = 0
    total_logistic_loss = 0.0
    total_edges = 0

    bce_loss = torch.nn.BCELoss()  # Logistic loss 계산
    
    save_dir="GNN_visualized_graphs"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)

            # 모델 예측
            edge_pred = model(data.x, data.edge_index, data.edge_attr)

            # Sigmoid를 통해 예측 확률로 변환
            edge_pred_sigmoid = torch.sigmoid(edge_pred)

            # Binary Prediction (0.5 임계값)
            binary_preds = (edge_pred_sigmoid >= 0.5).float()

            # 정답 라벨
            edge_labels = data.edge_attr[:, 0]  # 최적 경로 포함 여부

            # Precision, Recall 계산용
            total_correct += (binary_preds * edge_labels).sum().item()  # True Positive
            total_positive += edge_labels.sum().item()  # Ground Truth Positive
            total_predicted_positive += binary_preds.sum().item()  # Predicted Positive

            # Logistic Loss 계산
            total_logistic_loss += bce_loss(edge_pred_sigmoid, edge_labels).item() * edge_labels.size(0)

            # 전체 엣지 수
            total_edges += edge_labels.size(0)

            # 시각화
            if batch_idx < 5:  # 처음 5개 배치만 시각화
                save_path = os.path.join(save_dir, f"tsp_graph_{batch_idx + 1}.png")
                visualize_tsp(data, edge_pred_sigmoid, binary_preds, edge_labels, title=f"TSP Graph {batch_idx + 1}", save_path=save_path)

    # Precision, Recall 계산
    precision = total_correct / total_predicted_positive if total_predicted_positive > 0 else 0.0
    recall = total_correct / total_positive if total_positive > 0 else 0.0

    # Average Logistic Loss
    avg_logistic_loss = total_logistic_loss / total_edges if total_edges > 0 else 0.0

    print(f"Optimal Edges Precision: {precision * 100:.5f}%")
    print(f"Optimal Edges Recall: {recall * 100:.5f}%")
    print(f"Average Logistic Loss: {avg_logistic_loss:.10f}")

# 모델 평가 및 시각화 실행
evaluate_model_and_visualize(model, test_loader)
