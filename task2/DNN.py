import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class DNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.layer2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.layer3 = nn.Linear(32, 16)
        self.dropout3 = nn.Dropout(p=dropout_rate)

        self.output = nn.Linear(16, 1) 

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout1(x)
        x = self.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.relu(self.layer3(x))
        x = self.dropout3(x)
        x = self.sigmoid(self.output(x)) 
        return x
