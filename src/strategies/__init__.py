from flwr.server.strategy import FedAvg, FedProx, FedAdam
from .dynamic import DynamicLayerStrategy
from .fednova import FedNova

def get_strategy(strategy_name: str, **kwargs):
    """
    Factory to get the requested strategy.
    """
    name = strategy_name.lower()
    
    # --- FIX: EXTRACT DYNAMIC ARGS ---
    # We remove these from kwargs so they don't break FedAvg/FedNova/etc.
    start_clients = kwargs.pop('start_clients', 2)
    end_clients = kwargs.pop('end_clients', 5)
    switch_round = kwargs.pop('switch_round', 2)
    
    if name == "dynamic":
        # Pass them explicitly to Dynamic strategy
        return DynamicLayerStrategy(
            start_clients=start_clients,
            end_clients=end_clients,
            switch_round=switch_round,
            **kwargs
        )
        
    elif name == "fednova":
        return FedNova(**kwargs)
        
    elif name == "fedavg":
        return FedAvg(**kwargs)
        
    elif name == "fedprox":
        if 'proximal_mu' not in kwargs:
            kwargs['proximal_mu'] = 0.1
        return FedProx(**kwargs)
        
    elif name == "fedadam":
        return FedAdam(**kwargs)
        
    else:
        raise ValueError(f"Strategy '{strategy_name}' not found in src/strategies/")
