import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
from tqdm import tqdm



# 모델 정의
class TSPFNN(nn.Module):
    def __init__(self):
        super(TSPFNN, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # 입력: (from_x, from_y, to_x, to_y)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)  # 출력: 최적 경로 포함 확률
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # 확률 출력
        return x


class TSPDataset(Dataset):
    def __init__(self, folder_path):
        # 모든 데이터를 하나의 DataFrame으로 통합
        self.data = pd.concat([pd.read_csv(f"{folder_path}/tsp_data_{i}.csv") for i in range(1, 501)])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = torch.tensor([row['from_x'], row['from_y'], row['to_x'], row['to_y']], dtype=torch.float32)
        y = torch.tensor(row['is_in_tour'], dtype=torch.float32)
        return x, y
    
    
print('zone 1')
# 모델 인스턴스 생성
model = TSPFNN()
criterion = nn.BCELoss()  # 이진 분류 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)



print('zone 2')
# 데이터 로더 설정
start_time = time.time()
dataset = TSPDataset('dataset')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

print('zone 3')
# 학습 루프 예시
for epoch in range(10):
    running_loss = 0.0
    # tqdm으로 배치 진행률 표시
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{10}")):
        optimizer.zero_grad()
        output = model(x_batch).squeeze()  # 예측 확률
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/10] completed, Average Loss: {running_loss / len(train_loader):.4f}")
end_time = time.time()

torch.save(model.state_dict(), 'tsp_fnn_model.pth')