from logging import INFO
from typing import List, Tuple
from flwr.common import Metrics, Parameters, FitIns, log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

class DynamicLayerStrategy(FedAvg):
    def __init__(self, start_clients=2, end_clients=5, switch_round=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_clients = start_clients
        self.end_clients = end_clients
        self.switch_round = switch_round

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        if server_round <= self.switch_round:
            current = self.start_clients
        else:
            current = self.end_clients
            
        log(INFO, f"[Dynamic Strategy] Round {server_round}: Requesting {current} clients.")
        
        self.min_fit_clients = current
        self.min_evaluate_clients = current
        self.min_available_clients = current
        self.fraction_fit = 0.0
        self.fraction_evaluate = 0.0
        
        return super().configure_fit(server_round, parameters, client_manager)
