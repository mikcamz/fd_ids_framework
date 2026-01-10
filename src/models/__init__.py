from .cnn_lstm import CnnLstmNet
from .mlp import SimpleMLP

def get_model(model_name: str, input_dim: int, num_classes: int):
    name = model_name.lower()
    
    if name == "cnn_lstm":
        return CnnLstmNet(input_dim, num_classes)
    elif name == "mlp":
        return SimpleMLP(input_dim, num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not found in src/models/")
