from torch import nn, optim

BATCH_SIZE = 32
LR = 0.001
EPOCH = 200
T_MAX = 50
ETA_MIN = 0.00001
STEP_SIZE = 50
GAMMA = 0.5
criterion = nn.CrossEntropyLoss()
new_model_train = True
model_type = "resnet50"
architecture = "CNN"
dataset = "aircraft"
save_model_path = f"./data/model/{model_type}_{dataset}.pt"
save_graph_path = f"./results/graph/{model_type}_{dataset}_train_val_loss.jpg"
save_test_graph_path = f"./results/graph/{model_type}_{dataset}_test_plot.jpg"