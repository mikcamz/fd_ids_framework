import os
import json
from typing import List, Tuple
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context, ndarrays_to_parameters, Metrics

from .models import get_model
from .strategies import get_strategy

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregates metrics and PRINTS them so you see them every round.
    """
    # Extract values
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    total_examples = sum(examples)
    
    # Calculate weighted averages
    agg_accuracy = sum(accuracies) / total_examples
    agg_f1 = sum(f1_scores) / total_examples
    
    # --- NEW: Print to console immediately ---
    print(f"\n" + "="*40)
    print(f"  ~~~ ROUND METRICS AGGREGATION ~~~")
    print(f"  Clients Reported: {len(metrics)}")
    print(f"  Avg Accuracy:     {agg_accuracy:.4f} ({agg_accuracy*100:.2f}%)")
    print(f"  Avg F1-Score:     {agg_f1:.4f}")
    print(f"="*40 + "\n")
    # ----------------------------------------

    return {
        "accuracy": agg_accuracy,
        "f1_score": agg_f1,
    }

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
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)