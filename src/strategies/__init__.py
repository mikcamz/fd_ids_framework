from flwr.server.strategy import FedAvg, FedProx, FedAdam
from .fednova import FedNova
from .dynamic import DynamicFedAvg, DynamicFedNova

def get_strategy(strategy_name: str, **kwargs):
    name = strategy_name.lower()
    
    # Extract the schedule string
    schedule_json = kwargs.pop('schedule_json', "[]")
    
    # If using standard strategies, we check if we should "upgrade" them to Dynamic
    # We assume we always want dynamic logic if the user provided a schedule in config.
    
    if name == "fedavg":
        # Wrap FedAvg with Dynamic Logic
        return DynamicFedAvg(schedule_json=schedule_json, **kwargs)

    elif name == "fednova":
        # Wrap FedNova with Dynamic Logic
        return DynamicFedNova(schedule_json=schedule_json, **kwargs)
        
    else:
        # Fallback for simple testing
        return FedAvg(**kwargs)