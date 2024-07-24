import os
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Normalize, AutoAugment, AutoAugmentPolicy
from torch.utils.data import Dataset, DataLoader
from custom_dataset import Custom_Dataset
from hyper_parameters import BATCH_SIZE
from cutmix import CutMixCollator
from mixup import MixUpCollator

image_dir = "./data/images"
train_csv = "./csv/train.csv"
test_csv = "./csv/test.csv"
val_csv = "./csv/val.csv"

class RemoveBottomPixels(object):
    def __init__(self, pixels_to_remove=20):
        self.pixels_to_remove = pixels_to_remove

    def __call__(self, img):
        width, height = img.size
        return img.crop((0, 0, width, height - self.pixels_to_remove))

class SobelFilter(object):
    def __init__(self, dx=1, dy=1, ksize=3, scale=1, delta=0):
        self.dx = dx
        self.dy = dy
        self.ksize = ksize
        self.scale = scale
        self.delta = delta

    def __call__(self, img):
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobel = cv2.Sobel(src=gray, ddepth=-1, dx=self.dx, dy=self.dy, ksize=self.ksize, scale=self.scale, delta=self.delta)
        sobel_rgb = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(sobel_rgb)

# alpha = 1.0

# 트랜스폼
transform_train = transforms.Compose([
    RemoveBottomPixels(pixels_to_remove=20),
    transforms.RandAugment(),
    # AutoAugment(policy=AutoAugmentPolicy.IMAGENET,interpolation=transforms.InterpolationMode.BILINEAR),
    # transforms.AugMix(),
    # transforms.RandomApply([SobelFilter(dx=1, dy=1, ksize=5, scale=3, delta=0)], p=0.5),
    transforms.Resize((224, 224)),
    # # transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.5, hue=0.15)], p=0.5),
    # transforms.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.15)),  
    # transforms.RandomRotation(degrees=15),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    # transforms.RandomErasing(p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_val = transforms.Compose([
    RemoveBottomPixels(pixels_to_remove=20),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_test = transforms.Compose([
    RemoveBottomPixels(pixels_to_remove=20),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 데이터셋
train_DS = Custom_Dataset(csv_file=train_csv, root_dir=image_dir, transform=transform_train)
test_DS = Custom_Dataset(csv_file=test_csv, root_dir=image_dir, transform=transform_test)
val_DS = Custom_Dataset(csv_file=val_csv, root_dir=image_dir, transform=transform_val)

# cutmix_collator = CutMixCollator(alpha=alpha)
# mixup_collator = MixUpCollator(alpha=1.0)

# 데이터로더
train_DL = DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
# train_DL = DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True, collate_fn=mixup_collator)
# train_DL = DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True, collate_fn=cutmix_collator)
test_DL = DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=False)
val_DL = DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=False)