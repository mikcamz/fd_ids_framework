import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=20, num_classes=34):
        super(SimpleMLP, self).__init__()
        # Flatten input: 1 channel * 20 features
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Flatten: (Batch, 1, 20) -> (Batch, 20)
        x = x.view(x.shape[0], -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
