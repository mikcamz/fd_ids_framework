import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnLstmNet(nn.Module):
    def __init__(self, input_dim=20, num_classes=34):
        super(CnnLstmNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        _, (hn, _) = self.lstm2(x)
        x = hn[-1]
        x = self.dropout2(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
