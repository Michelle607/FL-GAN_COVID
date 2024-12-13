import os
import time
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report
import numpy as np

# 기본 설정
image_size = 224  # B7은 더 큰 입력 크기를 권장하지만, 동일 크기로 맞춥니다.
batch_size = 8
epochs = 10
learning_rate = 0.001

# 데이터 경로 설정
train_dir = "E:/downloads/PycharmProjects_E/FL-GAN_COVID/CovidDataset/train"
test_dir = "E:/downloads/PycharmProjects_E/FL-GAN_COVID/CovidDataset/test"

# 데이터 전처리
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 데이터셋 로드
train_dataset = ImageFolder(root=train_dir, transform=train_transform)
test_dataset = ImageFolder(root=test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# EfficientNet 모델 정의 (B0, B7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = [
    EfficientNet.from_pretrained('efficientnet-b0', num_classes=3).to(device),
    EfficientNet.from_pretrained('efficientnet-b7', num_classes=3).to(device)
]

# 손실 함수 및 옵티마이저 설정
criterions = [nn.CrossEntropyLoss() for _ in models]
optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in models]

# 클래스 이름 가져오기
class_names = list(train_dataset.class_to_idx.keys())

# 모델 학습
for epoch in range(epochs):
    for model in models:
        model.train()

    running_loss = [0.0] * len(models)
    correct = [0] * len(models)
    total = 0
    all_labels = []
    start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())

        for i, model in enumerate(models):
            optimizers[i].zero_grad()
            outputs = model(inputs)
            loss = criterions[i](outputs, labels)
            loss.backward()
            optimizers[i].step()

            running_loss[i] += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct[i] += (predicted == labels).sum().item()

        elapsed_time = time.time() - start_time
        batches_left = len(train_loader) - (batch_idx + 1)
        eta = batches_left * (elapsed_time / (batch_idx + 1))
        print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], ETA: {eta:.2f} sec",
              end='\r')

    # 각 모델별 Metrics 계산
    for i in range(len(models)):
        accuracy = 100 * correct[i] / total
        print(f"\nModel-{i + 1}: Epoch [{epoch + 1}/{epochs}], "
              f"Loss: {running_loss[i] / len(train_loader):.4f}, "
              f"Accuracy: {accuracy:.2f}%")

# 모델 평가 - 앙상블 (Soft Voting)
correct = 0
total = 0
all_preds = []
all_labels = []

for model in models:
    model.eval()

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 모든 모델의 확률 예측 수집
        ensemble_outputs = [torch.softmax(model(inputs), dim=1) for model in models]

        # Soft Voting: 확률 평균
        avg_outputs = torch.mean(torch.stack(ensemble_outputs), dim=0)

        _, predicted = torch.max(avg_outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 최종 Metrics 계산
test_accuracy = 100 * correct / total
classification_metrics = classification_report(all_labels, all_preds, target_names=class_names)

print("\nTest Results:")
print(f"Accuracy: {test_accuracy:.2f}%")
print(classification_metrics)
