from typing import Dict, List, Optional, Tuple, Union
from flwr.server.strategy import FedProx as FlwrFedProx
from flwr.common import Parameters, Scalar

class FedProx(FlwrFedProx):
    """
    FedProx Strategy wrapper.
    
    FedProx adds a proximal term to the local objective function to limit
    client drift (how far weights move from the global model).
    
    Key Param:
        proximal_mu (float): The weight of the proximal term. 
                             Higher = restrict clients more (good for Non-IID).
                             Lower = allow more freedom (closer to FedAvg).
    """
    def __init__(self, proximal_mu: float = 0.1, *args, **kwargs):
        # Pass the mu param to the parent Flower class
        super().__init__(proximal_mu=proximal_mu, *args, **kwargs)

    def __repr__(self) -> str:
        return f"FedProx(proximal_mu={self.proximal_mu})"