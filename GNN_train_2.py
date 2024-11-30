import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch.utils.data import Dataset
import os
import time
from tqdm import tqdm

data_folder = "pt_dataset"

# GNN 모델 정의
class TSPGNN(nn.Module):
    def __init__(self):
        super(TSPGNN, self).__init__()
        self.conv1 = GCNConv(2, 64, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(64, 32, add_self_loops=True, normalize=True)
        self.edge_mlp = nn.Sequential(
            nn.Linear(66, 16),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout 추가
            nn.Linear(16, 1)  # 엣지가 최적 경로에 포함될 확률 예측
        )

    def forward(self, x, edge_index, edge_attr):
        # 노드 임베딩
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        # 엣지 특성 생성
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=1)

        # 강제 크기 조정 (필요시)
        if edge_features.size(1) != 66:
            edge_features = torch.cat(
                [edge_features, torch.zeros(edge_features.size(0), 66 - edge_features.size(1)).to(edge_features.device)],
                dim=1
            )

        # 엣지 예측
        edge_output = self.edge_mlp(edge_features)  # logits 출력
        return edge_output.squeeze()


class TSPGraphDataset(Dataset):
    def __init__(self, folder_path=data_folder):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in the specified folder: {folder_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        graph_data = torch.load(self.files[idx])
        return graph_data


if __name__ == "__main__":
    # 모델 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TSPGNN().to(device)

    # 손실 함수 및 옵티마이저 설정
    positive_weight = 10.0  # 데이터 불균형 보정
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 학습률 감소
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # 학습률 스케줄러

    # 데이터 로더 설정
    dataset = TSPGraphDataset(data_folder)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 학습 루프
    num_epochs = 10
    start_time = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        # tqdm으로 진행률 표시
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            edge_pred_logits = model(data.x, data.edge_index, data.edge_attr)

            # 엣지의 최적 경로 포함 여부 라벨
            edge_labels = data.edge_attr[:, 0]

            # 손실 계산
            loss = bce_loss(edge_pred_logits, edge_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()  # 학습률 업데이트
        print(f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {running_loss / len(train_loader):.4f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    # 모델 저장
    torch.save(model.state_dict(), 'larger_tsp_gnn_edge_model.pth')
