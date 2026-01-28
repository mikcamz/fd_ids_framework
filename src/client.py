import os
import torch
import numpy as np
from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Context
from collections import OrderedDict

# Import metrics
from sklearn.metrics import f1_score
from src.models import get_model
from src.dataset import load_client_data

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net, trainloader, epochs, lr=0.001):
    """
    Trains the network and calculates training metrics (Accuracy, F1, Loss).
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    
    # Initialize accumulators for metrics
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # --- RESEARCH METRICS COLLECTION ---
            # Accumulate loss (multiply by batch size to weight correctly)
            running_loss += loss.item() * labels.size(0)
            
            # Collect predictions for Acc/F1
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate final metrics over the entire local training run
    total_samples = len(trainloader.dataset) * epochs
    avg_loss = running_loss / total_samples
    
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, acc, f1

def test(net, testloader):
    """
    Evaluates the network on the test set.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            
            # Weighted loss calculation
            running_loss += criterion(outputs, labels).item() * labels.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    if len(testloader.dataset) == 0: 
        return 0.0, 0.0, 0.0
    
    avg_loss = running_loss / len(testloader.dataset)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, acc, f1

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Train and capture REAL metrics
        loss, acc, f1 = train(self.net, self.trainloader, self.local_epochs)
        
        steps = self.local_epochs * len(self.trainloader)
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "train_loss": loss,
            "tau": steps,
            # --- REAL METRICS FOR RESEARCH ---
            "accuracy": float(acc), 
            "f1_score": float(f1)
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc, f1 = test(self.net, self.testloader)
        return float(loss), len(self.testloader.dataset), {
            "accuracy": float(acc), 
            "f1_score": float(f1)
        }

def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    
    # Read Env Vars
    model_name = os.environ.get("FLWR_MODEL_NAME", "cnn_lstm")
    data_dir = os.environ.get("DATA_DIR", "data/")
    
    # Load Data & Model
    trainloader, testloader = load_client_data(partition_id, data_dir, batch_size=32)
    net = get_model(model_name, input_dim=20, num_classes=34).to(DEVICE)
    
    return FlowerClient(net, trainloader, testloader, local_epochs=1).to_client()

app = ClientApp(client_fn=client_fn)