import torch
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
from GNN_train import *
from torch.nn.functional import binary_cross_entropy_with_logits  # 로지스틱 손실 함수

# 모델 정의 및 가중치 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TSPGNN().to(device)
model.load_state_dict(torch.load('tsp_gnn_model.pth', map_location=device))
model.eval()

# 테스트 데이터셋 정의
class TSPTestDataset(Dataset):
    def __init__(self, folder_path='pt_dataset_test'):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in the specified folder: {folder_path}")
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        graph_data = torch.load(self.files[idx])
        return graph_data


# 테스트 데이터 로더 설정
test_dataset = TSPTestDataset('pt_dataset_test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 모델 평가 함수: logit loss 기반
def evaluate_model(model, test_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # 예측 수행 (로짓으로 출력됨)
            edge_pred = model(data.x, data.edge_index, data.edge_attr)
            loss = binary_cross_entropy_with_logits(edge_pred.squeeze(), data.edge_attr.squeeze())
            
            # 손실 및 정확도 계산
            total_loss += loss.item() * data.edge_attr.size(0)
            edge_prob = torch.sigmoid(edge_pred)  # 로짓을 확률로 변환
            binary_preds = (edge_prob >= 0.5).float()
            
            # 정확도 계산
            total_correct += (binary_preds == data.edge_attr).sum().item()
            total_samples += data.edge_attr.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f"Average Logistic Loss on Test Set: {avg_loss:.4f}")
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# 모델 평가
evaluate_model(model, test_loader)