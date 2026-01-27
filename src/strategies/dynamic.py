import json
from logging import INFO
from typing import List, Tuple, Dict
from flwr.common import Parameters, FitIns, log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from .fednova import FedNova

class DynamicLogicMixin:
    """
    Parses a schedule list and determines how many clients to run for the current round.
    Schedule format: [{"round_start": 1, "active_clients": 2}, ...]
    """
    def __init__(self, schedule_json: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parse the JSON string back into a list
        self.schedule = json.loads(schedule_json)
        # Sort by round_start to ensure correct logic
        self.schedule.sort(key=lambda x: x['round_start'])

    def _get_client_count(self, round_num: int) -> int:
        """Finds the active client count for the current round."""
        selected_count = 2 # Default fallback
        
        for stage in self.schedule:
            if round_num >= stage['round_start']:
                selected_count = stage['active_clients']
            else:
                # Since list is sorted, once we pass the current round, stop.
                break
        return selected_count

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        # 1. Determine clients for this round
        current_clients = self._get_client_count(server_round)
            
        log(INFO, f"[Dynamic Strategy] Round {server_round}: Scheduling {current_clients} clients.")
        
        # 2. Force Flower settings
        self.min_fit_clients = current_clients
        self.min_evaluate_clients = current_clients
        self.min_available_clients = current_clients
        self.fraction_fit = 0.0
        self.fraction_evaluate = 0.0
        
        return super().configure_fit(server_round, parameters, client_manager)

# Combined Classes
class DynamicFedAvg(DynamicLogicMixin, FedAvg): pass
class DynamicFedNova(DynamicLogicMixin, FedNova): pass