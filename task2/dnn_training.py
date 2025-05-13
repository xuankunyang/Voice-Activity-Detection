import torch
from DNN import DNN
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from audio_file_process import process_audio_file_traning
import numpy as np
from pathlib import Path
from vad_utils import read_label_from_file
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from evaluate import get_metrics
import json
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist() 
        return super().default(o)

def dnn_training(frame_length: int, frame_step: int, num_epochs: int = 10, dropout_rate: float = 0.5, lr:float = 0.001, batch_size:int = 32):

    input_dim = 69 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model = DNN(input_dim, dropout_rate).to(device)

    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    data = np.load(f'task2/training_data_features_and_labels/length_{frame_length}_step_{frame_step}.npz')

    X = data['features']
    y = data['labels']

    # 使用相同的数据集进行训练和验证
    # 训练集和验证集的划分
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 将数据转换为Tensor并构建DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device) 
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device) 

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device) 
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device) 

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 不需要在验证集上进行shuffle

    # 初始化日志字典
    training_log = {
        "train": [],
        "validation": [],
        "params": []
    }

    training_log["params"].append({
            "Parameters:": "----------------------Parameters----------------------",
            "lr:": lr,
            "Optimizier:": 'Adam',
            "num_epochs:": num_epochs, 
            "batch_size": batch_size
        })

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) 

            optimizer.zero_grad()  # 清空梯度

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            train_loss += loss.item()

            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            total_train += labels.size(0)

        train_acc = train_correct / total_train
        avg_train_loss = train_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # 保存训练集的日志
        training_log["train"].append({
            "training:": "----------------------Training----------------------",
            "epoch:": epoch + 1,
            "loss:": avg_train_loss,
            "accuracy:": train_acc
        })

        # 验证集评估
        model.eval() 
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                total_val += labels.size(0)

        val_acc = val_correct / total_val
        avg_val_loss = val_loss / len(val_loader)

        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # 保存验证集的日志
        training_log["validation"].append({
            "Validating:": "----------------------Validating----------------------",
            "epoch:": epoch + 1,
            "loss:": avg_val_loss,
            "accuracy:": val_acc
        })

    log_dir = 'task2/logs'
    log_file = os.path.join(log_dir, f'DNN_experiment_log_for_frame_length_{frame_length}_step_{frame_step}.json')
    with open(log_file, 'a') as f:
        json.dump(training_log, f, indent=4, cls=NumpyEncoder)
        f.write('\n')  # 每次日志记录一行

    print(f"Training log saved to {log_file}")

    file_path = f'task2/models/DNN_for_frame_length_{frame_length}_step_{frame_step}.pth'
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")



