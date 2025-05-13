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

def dnn_testing(frame_length: int, frame_step: int):

    input_dim = 69 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    model = DNN(input_dim).to(device)
    model_path = f'task2/models/DNN_for_frame_length_{frame_length}_step_{frame_step}.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval() 

    data = np.load(f'task2/dev_features_and_labels/length_{frame_length}_step_{frame_step}.npz')
    X_test = data['features']
    y_test = data['labels']

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    testing_log = {
        "test": []
    }

    test_loss = 0.0
    test_correct = 0
    total_test = 0
    all_preds = []
    all_labels = []
    all_probs = []

    criterion = nn.BCELoss() 

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            predictions = (outputs > 0.5).float()
            test_correct += (predictions == labels).sum().item()
            total_test += labels.size(0)
            all_probs.append(outputs.cpu().numpy())
            
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    test_acc = test_correct / total_test
    avg_test_loss = test_loss / len(test_loader)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    auc, eer = get_metrics(prediction=all_probs, label=all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    confusion = confusion_matrix(all_labels, all_preds)

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion}")

    testing_log["test"].append({
        "Testing": "----------------------Testing----------------------",
        "loss:": avg_test_loss,
        "AUC:": auc, 
        "EER:": eer, 
        "accuracy:": accuracy,
        "precision:": precision,
        "recall:": recall,
        "f1_score:": f1,
        "confusion_matrix:": confusion.tolist() 
    })

    # 保存测试日志到JSON文件
    # log_dir = 'task2/logs'
    # log_file = os.path.join(log_dir, f'DNN_testing_log_for_frame_length_{frame_length}_step_{frame_step}.json')
    # with open(log_file, 'a') as f:
    #     json.dump(testing_log, f, indent=4, cls=NumpyEncoder)
    #     f.write('\n')  # 每次日志记录一行

    # print(f"Testing log saved to {log_file}")


    return y_test, all_probs