import os
import pandas as pd
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_ranking(y_hat, y_batch):
    pred_top1 = y_hat.argmax(dim=1, keepdim=True)
    correct_top1 = pred_top1.eq(y_batch.view_as(pred_top1)).sum().item()

    _, pred_top5 = y_hat.topk(5, dim=1, largest=True, sorted=True)
    correct_top5 = pred_top5.eq(y_batch.view(-1, 1).expand_as(pred_top5)).sum().item()

    total = y_batch.size(0)
    top1_error = 1 - (correct_top1 / total)
    top5_error = 1 - (correct_top5 / total)

    return top1_error, top5_error

def Train(model, train_DL, criterion, optimizer, class_correct_counts, class_total_counts):
    model.train()
    N = len(train_DL.dataset)
    rloss = 0
    rcorrect = 0
    total_top1 = 0
    total_top5 = 0

    for x_batch, y_batch in train_DL:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        y_hat = model(x_batch)
        loss = criterion(y_hat, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_b = loss.item() * x_batch.shape[0]
        rloss += loss_b
        
        correct_top1, correct_top5 = calculate_ranking(y_hat, y_batch)
        total_top1 += correct_top1 * x_batch.size(0)
        total_top5 += correct_top5 * x_batch.size(0)
        
        # 클래스별로 정답 개수와 총 개수를 추적
        _, predicted_classes = torch.max(y_hat, 1)
        for i in range(len(y_batch)):
            if predicted_classes[i] == y_batch[i]:
                class_correct_counts[y_batch[i]] += 1
            class_total_counts[y_batch[i]] += 1

    loss_e = rloss / N
    top1_error_e = total_top1 / N * 100
    top5_error_e = total_top5 / N * 100
    accuracy_e = 100 - top1_error_e
    
    return loss_e, accuracy_e, top1_error_e, top5_error_e, class_correct_counts, class_total_counts

def Validate(model, val_DL, criterion, class_correct_counts, class_total_counts):
    model.eval()
    with torch.no_grad():
        N = len(val_DL.dataset)
        rloss = 0
        rcorrect = 0
        total_top1 = 0
        total_top5 = 0
        
        for x_batch, y_batch in val_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            loss_b = loss.item() * x_batch.shape[0]
            rloss += loss_b
            
            correct_top1, correct_top5 = calculate_ranking(y_hat, y_batch)
            total_top1 += correct_top1 * x_batch.size(0)
            total_top5 += correct_top5 * x_batch.size(0)
            
            # 클래스별로 정답 개수와 총 개수를 추적
            _, predicted_classes = torch.max(y_hat, 1)
            for i in range(len(y_batch)):
                if predicted_classes[i] == y_batch[i]:
                    class_correct_counts[y_batch[i]] += 1
                class_total_counts[y_batch[i]] += 1

        loss_e = rloss / N
        top1_error_e = total_top1 / N * 100
        top5_error_e = total_top5 / N * 100
        accuracy_e = 100 - top1_error_e

    return loss_e, accuracy_e, top1_error_e, top5_error_e, class_correct_counts, class_total_counts

def Test(model, test_DL):
    model.eval()
    with torch.no_grad():
        N = len(test_DL.dataset)
        rcorrect = 0
        total_top1 = 0
        total_top5 = 0
        
        for x_batch, y_batch in test_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            y_hat = model(x_batch)
            
            correct_top1, correct_top5 = calculate_ranking(y_hat, y_batch)
            total_top1 += correct_top1 * x_batch.size(0)
            total_top5 += correct_top5 * x_batch.size(0)
            
        top1_error_e = total_top1 / N * 100
        top5_error_e = total_top5 / N * 100
        accuracy_e = 100 - top1_error_e

    return accuracy_e, top1_error_e, top5_error_e

def train_val_graph(train_loss_history, val_loss_history, save_graph_path):
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Train/Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(save_graph_path)
    plt.close()

def Test_plot(model, test_DL, save_graph_path):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_DL))
        x_batch = x_batch.to(DEVICE)
        y_hat = model(x_batch)
        pred = y_hat.argmax(dim=1)
        
    x_batch = x_batch.to("cpu")
    
    x_batch = torch.clamp(x_batch, 0, 1)

    plt.figure(figsize=(8,4))
    for idx in range(6):
        plt.subplot(2,3, idx+1, xticks=[], yticks=[])
        plt.imshow(x_batch[idx].permute(1,2,0), cmap="gray")
        pred_class = test_DL.dataset.classes[pred[idx]]
        true_class = test_DL.dataset.classes[y_batch[idx]]
        plt.title(f"{pred_class} ({true_class})", color = "g" if pred_class==true_class else "r")

    plt.savefig(save_graph_path)
    plt.close()

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num
