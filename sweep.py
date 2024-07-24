import wandb
import torch
import yaml
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import resnet50
from functions import Train, Validate, Test, Test_plot
from data_loader import train_DL, val_DL, test_DL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def sweep():
    wandb.init()
    
    config = wandb.config

    model = resnet50(num_classes=100).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN)
    
    train_loss_history = []
    val_loss_history = []
    early_stopping_epochs = 20
    best_loss = float('inf')
    early_stop_counter = 0
    save_model_path = "best_model.pth"
    class_correct_counts = torch.zeros(100)  # Assuming 100 classes
    class_total_counts = torch.zeros(100)  # Assuming 100 classes

    for epoch in range(config.EPOCH):
        train_loss, train_acc, train_top1, train_top5, class_correct_counts, class_total_counts = Train(model, train_DL, criterion, optimizer, class_correct_counts, class_total_counts)
        val_loss, val_acc, val_top1, val_top5, class_correct_counts, class_total_counts = Validate(model, val_DL, criterion, class_correct_counts, class_total_counts)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch: {epoch+1}, Current LR = {current_lr}")
        print(f"Epoch [{epoch+1}/{config.EPOCH}] - Train Loss: {train_loss:.4f} - Train Accuracy: {train_acc:.2f}% - Train Top1 Error: {train_top1:.2f}% - Train Top5 Error: {train_top5:.2f}%")    
        print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.2f}% - Val Top1 Error: {val_top1:.2f}% - Val Top5 Error: {val_top5:.2f}%")
        print("-" * 20)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        wandb.log({
            "Train Loss": train_loss,
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
            torch.save(model, save_model_path)
            best_loss = val_loss
            early_stop_counter = 0

            # Save the best metrics
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

            # Calculate and save class accuracies
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

if __name__ == "__main__":
    with open('sweep.yaml', 'r') as file:
        sweep_config = yaml.safe_load(file)
    
    sweep_id = wandb.sweep(sweep_config, project="aircraft")
    wandb.agent(sweep_id, function=sweep, count=20)
