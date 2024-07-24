import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import cv2
from torchvision import transforms
from data_loader import test_DL
from model import resnet50  # ResNet 모델이 정의된 파일 이름을 resnet_model로 가정

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 결과 저장 디렉토리 생성
save_dir = "./results/feature_map"
os.makedirs(save_dir, exist_ok=True)

model_path = "data/model/resnet50_aircraft_8.pt"
load_model = torch.load(model_path, map_location=DEVICE)

# 모델 로드
load_model = load_model.to(DEVICE)
load_model.eval()

with torch.no_grad():  # 그라디언트 업데이트를 하지 않음
    x_batch, y_batch = next(iter(test_DL))  # 배치 하나 가져옴
    x_batch = x_batch.to(DEVICE)
    y_batch = y_batch.to(DEVICE)

    y_hat = load_model(x_batch)  # 결과는 아래 사진처럼 나올 것임
    pred = y_hat.argmax(dim=1)  # 10종 분류 중 가장 큰 값을 pred값으로 사용

    # 각 스테이지의 feature map 추출
    x = load_model.conv1(x_batch)
    x = load_model.bn1(x)
    x = load_model.relu(x)
    x = load_model.maxpool(x)

    feature_map1 = load_model.stage1(x)
    feature_map2 = load_model.stage2(feature_map1)
    feature_map3 = load_model.stage3(feature_map2)
    feature_map4 = load_model.stage4(feature_map3)

x_batch = x_batch.cpu()  # 그림 그릴 때는 cpu 사용해도 됨
feature_map1 = feature_map1.cpu()
feature_map2 = feature_map2.cpu()
feature_map3 = feature_map3.cpu()
feature_map4 = feature_map4.cpu()

# 원본 이미지 시각화 및 저장
plt.figure(figsize=(8, 8))
plt.xticks([]); plt.yticks([])
plt.imshow(x_batch[0, ...].permute(1, 2, 0))
plt.savefig(os.path.join(save_dir, 'original_image.png'))
plt.close()

# 첫 번째 스테이지의 feature map 시각화 및 저장
print(feature_map1.shape)
plt.figure(figsize=(32, 16))
for idx in range(32):
    plt.subplot(4, 8, idx + 1, xticks=[], yticks=[])
    plt.imshow(feature_map1[0, idx, ...], cmap="gray")
plt.savefig(os.path.join(save_dir, 'feature_map1.png'))
plt.close()

# 두 번째 스테이지의 feature map 시각화 및 저장
print(feature_map2.shape)
plt.figure(figsize=(16, 16))
for idx in range(64):
    plt.subplot(8, 8, idx + 1, xticks=[], yticks=[])
    plt.imshow(feature_map2[0, idx, ...], cmap="gray")
plt.savefig(os.path.join(save_dir, 'feature_map2.png'))
plt.close()

# 세 번째 스테이지의 feature map 시각화 및 저장
print(feature_map3.shape)
plt.figure(figsize=(16, 16))
for idx in range(128):
    plt.subplot(8, 16, idx + 1, xticks=[], yticks=[])
    plt.imshow(feature_map3[0, idx, ...], cmap="gray")
plt.savefig(os.path.join(save_dir, 'feature_map3.png'))
plt.close()

# 네 번째 스테이지의 feature map 시각화 및 저장
print(feature_map4.shape)
plt.figure(figsize=(16, 16))
for idx in range(256):
    plt.subplot(16, 16, idx + 1, xticks=[], yticks=[])
    plt.imshow(feature_map4[0, idx, ...], cmap="gray")
plt.savefig(os.path.join(save_dir, 'feature_map4.png'))
plt.close()

# 네 번째 스테이지의 feature map 합산 후 시각화 및 저장
summed_map = feature_map4.abs().sum(dim=1)
summed_map_resized = cv2.resize(summed_map[0].numpy(), (x_batch.shape[3], x_batch.shape[2]))

plt.figure(figsize=(8, 8))
plt.xticks([]); plt.yticks([])
plt.imshow(summed_map_resized, cmap="viridis")
plt.savefig(os.path.join(save_dir, 'summed_feature_map.png'))
plt.close()

# 원본 이미지와 합산된 feature map 오버레이 시각화 및 저장
plt.figure(figsize=(8, 8))
plt.xticks([]); plt.yticks([])
plt.imshow(x_batch[0, ...].permute(1, 2, 0))
plt.imshow(summed_map_resized, alpha=0.4, cmap='viridis')
pred_class = pred[0].item()
true_class = y_batch[0].item()
plt.title(f"{pred_class} ({true_class})", color="g" if pred_class == true_class else "r")
plt.savefig(os.path.join(save_dir, 'overlay_summed_feature_map.png'))
plt.close()

print("Feature maps and original image have been saved to 'results/feature_map'")
