import os
import time
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# 기본 설정
image_size = 64
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

# EfficientNet-B0 모델 정의
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)

# 모델 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 클래스 이름 가져오기
class_names = list(train_dataset.class_to_idx.keys())

# 모델 학습
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    start_time = time.time()  # 학습 시작 시간 기록

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # ETA 계산
        elapsed_time = time.time() - start_time
        batches_left = len(train_loader) - (batch_idx + 1)
        eta = batches_left * (elapsed_time / (batch_idx + 1))
        print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], ETA: {eta:.2f} sec", end='\r')

    # Metrics 계산
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(
        f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, "
        f"Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
    )

# 모델 평가
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 최종 Metrics 계산
test_accuracy = 100 * correct / total
classification_metrics = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

print("\nTest Results:")
print(f"Accuracy: {test_accuracy:.2f}%")
for cls_name in class_names:
    print(f"{cls_name}: Precision: {classification_metrics[cls_name]['precision']:.4f}, "
          f"Recall: {classification_metrics[cls_name]['recall']:.4f}, "
          f"F1 Score: {classification_metrics[cls_name]['f1-score']:.4f}")

