import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from model import GoodEdgeDistributionCNN

#x1, x2 = 50, 110
#y1, y2 = 25, 130

#input_tensor1 = torch.rand(1, 192, 192)  # RGB 이미지 가정 (채널 수 3)
#output_tensor1 = torch.rand(1, 192, 192)

class RandomDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 랜덤하게 1~191 사이의 서로 다른 두 정수 생성
        x1, x2 = sorted(torch.randint(0, 96, (2,)).tolist())
        x2 += 96
        y1, y2 = sorted(torch.randint(0, 96, (2,)).tolist())
        y2 += 96
        # input_tensor 생성 (모든 값은 0으로 초기화)
        input_tensor = torch.zeros(1, 192, 192)
        input_tensor[0, 0:192, 0:192] = 0.02
        input_tensor[0, x1, y1] = 2  # 좌상단 꼭짓점
        input_tensor[0, x1, y2] = 2  # 좌하단 꼭짓점
        input_tensor[0, x2, y1] = 2  # 우상단 꼭짓점
        input_tensor[0, x2, y2] = 2  # 우하단 꼭짓점

        # output_tensor 생성 (직사각형 영역 1, 나머지 0)
        output_tensor = torch.zeros(1, 192, 192)
        output_tensor[0, 0:192, 0:192] = 0.02
        output_tensor[0, x1:x1+1, y1:y2 + 1] = 2
        output_tensor[0, x2:x2+1, y1:y2 + 1] = 2
        output_tensor[0, x1:x2+1, y1:y1 + 1] = 2
        output_tensor[0, x1:x2+1, y2:y2 + 1] = 2
        
        #input_tensor = torch.rand(3, 192, 192)  # RGB 이미지 가정 (채널 수 3)
        #output_tensor = torch.rand(3, 192, 192)
        return input_tensor, output_tensor

# 데이터셋과 데이터 로더 설정
num_samples = 10009  # 예시로 1000개의 샘플 생성
batch_size = 8     # 배치 사이즈 설정

# 데이터셋 및 데이터 로더 생성
random_dataset = RandomDataset(num_samples=num_samples)
train_loader = DataLoader(random_dataset, batch_size=batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
class ScaledMSELoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(ScaledMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, predictions, targets):
        loss = self.mse_loss(predictions, targets)  # 기본 MSE 계산
        return loss * self.alpha  # alpha로 스케일링


model = GoodEdgeDistributionCNN().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = ScaledMSELoss(alpha=1)

# 데이터 로더 테스트
for param in model.parameters():
    param.requires_grad = True
model.train()
cnt = -1
for inputs, targets in train_loader:
    inputs = inputs.to(device)
    targets = targets.to(device)
    cnt += 1
    output = model(inputs)
    loss = criterion(output, targets)
    print(cnt, loss)
    optimizer.zero_grad()
    loss.backward()
    before_update = hash(str(model.state_dict()))
    optimizer.step()
    after_update = hash(str(model.state_dict()))
    truth = 0
    """for key in before_update:
        if not torch.equal(before_update[key], after_update[key]):
            truth = 1
            break
    if truth:
        print(f"Weights updated for layer: {key}")"""
    print(before_update, after_update)

    # 시각화를 위해 첫 번째 샘플 선택
    input_sample = inputs[0].cpu().squeeze(0).numpy()    # (192, 192)
    target_sample = targets[0].cpu().squeeze(0).numpy()  # (192, 192)
    output_sample = output[0].detach().cpu().squeeze(0).numpy()  # (192, 192)

    # 시각화
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(input_sample, cmap='gray')
    axs[0].set_title("Input")
    axs[1].imshow(target_sample, cmap='gray')
    axs[1].set_title("Target")
    axs[2].imshow(output_sample, cmap='gray')
    axs[2].set_title("Output")
    #if cnt % 1 == 0:
    plt.savefig('output_visualization' + str(cnt) + '.png')
    plt.close()
