import os
import pandas as pd
import numpy as np
import torch
import wandb
import optuna
import matplotlib.pyplot as plt

from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from model import resnet50
from functions import Train, Validate, Test, train_val_graph, Test_plot
from data_loader import train_DS, val_DS, test_DS, Custom_Dataset
# from cutmix import CutMixCollator, CutMixCriterion
from hyper_parameters import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

project_name = "aircraft"
model_dir = "./data/model"
results_path = "./results/history_path"

def get_new_index(dir_path, prefix):
    files = [f for f in os.listdir(dir_path) if f.startswith(prefix)]
    if not files:
        return 1
    indices = [int(f.split('_')[-1].split('.')[0]) for f in files if f.split('_')[-1].split('.')[0].isdigit()]
    return max(indices) + 1 if indices else 1

def objective(trial):
    # 하이퍼파라미터 제안
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    gamma = trial.suggest_float('gamma', 0.1, 0.9)
    step_size = trial.suggest_int('step_size', 10, 50)
    # alpha = trial.suggest_float('alpha', 0.1, 1.0)

    # 데이터로더 설정
    train_DL = DataLoader(train_DS, batch_size=batch_size, shuffle=True)
    val_DL = DataLoader(val_DS, batch_size=batch_size, shuffle=False)

    model = resnet50(num_classes=100).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    class_correct_counts = torch.zeros(100, dtype=torch.int32).to(DEVICE)
    class_total_counts = torch.zeros(100, dtype=torch.int32).to(DEVICE)

    best_val_loss = float('inf')

    for epoch in range(EPOCH):
        train_loss, train_acc, train_top1, train_top5, class_correct_counts, class_total_counts = Train(
            model, train_DL, criterion, optimizer, class_correct_counts, class_total_counts)
        val_loss, val_acc, val_top1, val_top5, class_correct_counts, class_total_counts = Validate(
            model, val_DL, criterion, class_correct_counts, class_total_counts)
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    return best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print(f"Best trial: {study.best_trial.value}")
    print("Best hyperparameters: ", study.best_trial.params)

    # 최적의 하이퍼파라미터를 사용하여 최종 모델 학습
    best_params = study.best_trial.params
    lr = best_params['lr']
    batch_size = best_params['batch_size']
    # alpha = best_params['alpha']
    gamma = best_params['gamma']
    step_size = best_params['step_size']

    train_DL = DataLoader(train_DS, batch_size=batch_size, shuffle=True)
    val_DL = DataLoader(val_DS, batch_size=batch_size, shuffle=False)
    test_DL = DataLoader(test_DS, batch_size=batch_size, shuffle=False)

    model_index = get_new_index(model_dir, f"{model_type}_{dataset}")
    save_model_path = f"{model_dir}/{model_type}_{dataset}_{model_index}.pt"

    wandb.init(project=project_name)

    model = resnet50(num_classes=100).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    train_loss_history = []
    val_loss_history = []
    early_stopping_epochs = 20
    best_loss = float('inf')
    early_stop_counter = 0

    class_correct_counts = torch.zeros(100, dtype=torch.int32).to(DEVICE)
    class_total_counts = torch.zeros(100, dtype=torch.int32).to(DEVICE)

    for epoch in range(EPOCH):
        train_loss, train_acc, train_top1, train_top5, class_correct_counts, class_total_counts = Train(
            model, train_DL, criterion, optimizer, class_correct_counts, class_total_counts)
        val_loss, val_acc, val_top1, val_top5, class_correct_counts, class_total_counts = Validate(
            model, val_DL, criterion, class_correct_counts, class_total_counts)
        
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch: {epoch+1}, Current LR = {current_lr}")
        print(f"Epoch [{epoch+1}/{EPOCH}] - Train Loss: {train_loss:.4f} - Train Accuracy: {train_acc:.2f}% - Train Top1 Error: {train_top1:.2f}% - Train Top5 Error: {train_top5:.2f}%")
        print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.2f}% - Val Top1 Error: {val_top1:.2f}% - Val Top5 Error: {val_top5:.2f}%")
        print("-" * 20)

        train_loss_history += [train_loss]
        val_loss_history += [val_loss]

        wandb.log({"Train Loss": train_loss,
                   "Val Loss": val_loss,
                   "Training accuracy": train_acc,
                   "Val accuracy": val_acc,
                   "Training top1 error": train_top1,
                   "Val top1 error": val_top1,
                   "Training top5 error": train_top5,
                   "Val top5 error": val_top5,
                   "Learning Rate": current_lr
                   })

        if val_loss < best_loss:
            torch.save(model.state_dict(), save_model_path)
            best_loss = val_loss
            early_stop_counter = 0

            # 최적의 val_loss일 때의 지표 저장
            best_metrics = {
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Training accuracy": train_acc,
                "Val accuracy": val_acc,
                "Training top1 error": train_top1,
                "Val top1 error": val_top1,
                "Training top5 error": train_top5,
                "Val top5 error": val_top5,
                "Learning Rate": current_lr
            }

            # 클래스별 정확도 계산
            class_accuracies = class_correct_counts.float() / class_total_counts.float() * 100
            sorted_indices = torch.argsort(class_accuracies, descending=True)
            top_5_classes = sorted_indices[:5].cpu().tolist()
            bottom_5_classes = sorted_indices[-5:].cpu().tolist()

            best_metrics["Top 5 Classes"] = [f"Class {i}: {class_accuracies[i].item():.2f}%" for i in top_5_classes]
            best_metrics["Bottom 5 Classes"] = [f"Class {i}: {class_accuracies[i].item():.2f}%" for i in bottom_5_classes]

        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stopping_epochs:
            print("Early Stopping!")
            break

        scheduler.step()

    # 최적 모델 로드
    model.load_state_dict(torch.load(save_model_path))

    # Test 실행
    test_acc, test_top1, test_top5 = Test(model, test_DL)

    print(f"Test accuracy: {test_acc:.2f}% - Test Top1 Error: {test_top1:.2f}% - Test Top5 Error: {test_top5:.2f}%")

    wandb.log({"Test Accuracy": test_acc,
               "Test Top1 Error": test_top1,
               "Test Top5 Error": test_top5
               })

    train_val_graph(train_loss_history, val_loss_history, save_graph_path)
    wandb.log({"Loss Graph": wandb.Image(save_graph_path)})

    Test_plot(model, test_DL, save_test_graph_path)
    wandb.log({"Test Plot": wandb.Image(save_test_graph_path)})

    wandb.finish()

    # 최적의 val_loss일 때의 지표를 CSV 파일로 저장
    metrics_index = get_new_index(results_path, f"{model_type}_{dataset}_metrics")
    csv_path = f"{results_path}/{model_type}_{dataset}_metrics_{metrics_index}.csv"
    metrics_df = pd.DataFrame([best_metrics])
    metrics_df.to_csv(csv_path, index=False)
    print(f"Best metrics saved to {csv_path}")