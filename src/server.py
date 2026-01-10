import os
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context, ndarrays_to_parameters
import torch

# Relative imports
from .models import get_model
from .strategies import get_strategy

def server_fn(context: Context):
    # 1. Read Config
    model_name = os.environ.get("FLWR_MODEL_NAME", "cnn_lstm")
    strategy_name = os.environ.get("FLWR_STRATEGY_NAME", "dynamic")
    num_rounds = int(os.environ.get("FLWR_ROUNDS", 5))
    
    # 2. Get Client Counts from Environment (Set by main.py)
    # Default to 2 if not set, to match your request
    fit_clients = int(os.environ.get("FLWR_FIT_CLIENTS", 2)) 

    # 3. Init Model
    print(f"[Server] Initializing Model: {model_name}")
    net = get_model(model_name, input_dim=20, num_classes=34)
    params = [val.cpu().numpy() for _, val in net.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(params)

    # 4. Init Strategy
    print(f"[Server] Strategy: {strategy_name} | Clients per round: {fit_clients}")
    
    strategy = get_strategy(
        strategy_name,
        initial_parameters=initial_parameters,
        
        # --- THE FIX ---
        # 1. Set fraction to 0.0 to disable percentage-based sampling
        fraction_fit=0.0,
        fraction_evaluate=0.0,
        
        # 2. Set the exact absolute number we want
        min_fit_clients=fit_clients,
        min_evaluate_clients=fit_clients,
        min_available_clients=fit_clients,
        
        # Dynamic strategy args (ignored by others)
        start_clients=3,
        end_clients=5,
        switch_round=3
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
