import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset 
import os
import time
from tqdm import tqdm


# GNN 모델 정의
class TSPGNN(nn.Module):
    def __init__(self):
        super(TSPGNN, self).__init__()
        self.conv1 = GCNConv(2, 64)  # 노드 입력: (x, y) 좌표
        self.conv2 = GCNConv(64, 32)
        self.edge_mlp = nn.Sequential(
            nn.Linear(32 * 2 + 1, 16), nn.ReLU(), nn.Linear(16, 1)
        )  # edge 예측을 위한 MLP

    def forward(self, x, edge_index, edge_attr):
        # 노드 임베딩 학습
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        # 엣지 특징 생성
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_prob = torch.sigmoid(self.edge_mlp(edge_features))  # 엣지 확률 예측
        return edge_prob
    
    
class TSPGraphDataset(Dataset):
    def __init__(self, folder_path='pt_dataset'):
        # 모든 .pt 파일 로드
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]
        
        # 폴더가 비어 있는지 확인
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in the specified folder: {folder_path}")
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # .pt 파일 로드
        graph_data = torch.load(self.files[idx])
        return graph_data


# 모델 인스턴스 생성
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TSPGNN().to(device)
criterion = nn.BCELoss()  # 이진 분류 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 데이터 로더 설정
dataset = TSPGraphDataset('pt_dataset')
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)  # 배치당 하나의 그래프

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
        edge_pred = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(edge_pred.squeeze(), data.edge_attr.squeeze())  # 실제 엣지 레이블과 비교하여 손실 계산
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {running_loss / len(train_loader):.4f}")

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# 모델 저장
torch.save(model.state_dict(), 'tsp_gnn_model.pth')