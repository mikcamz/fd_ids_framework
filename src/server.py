import os
import json
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context, ndarrays_to_parameters

from .models import get_model
from .strategies import get_strategy

def server_fn(context: Context):
    # 1. Read Env
    model_name = os.environ.get("FLWR_MODEL_NAME", "cnn_lstm")
    strategy_name = os.environ.get("FLWR_STRATEGY_NAME", "fedavg")
    num_rounds = int(os.environ.get("FLWR_ROUNDS", 5))
    schedule_json = os.environ.get("DYNAMIC_SCHEDULE", "[]")

    # 2. Init Model
    net = get_model(model_name, input_dim=20, num_classes=34)
    params = [val.cpu().numpy() for _, val in net.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(params)

    # 3. Init Strategy
    strategy = get_strategy(
        strategy_name,
        initial_parameters=initial_parameters,
        schedule_json=schedule_json, 
        min_available_clients=2      
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)