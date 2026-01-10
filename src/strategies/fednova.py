import numpy as np
from logging import INFO
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    log,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

class FedNova(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We need to track the global parameters state to calculate gradients (delta)
        self.global_parameters_arrays = None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        # Save the current global parameters (w_t) before sending them to clients
        # We need these to calculate the update vector: delta = w_t - w_i
        self.global_parameters_arrays = parameters_to_ndarrays(parameters)
        
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        # If we somehow missed saving global params, fallback to standard FedAvg
        if self.global_parameters_arrays is None:
            log(INFO, "[FedNova] Warning: Global parameters not found. Falling back to standard averaging.")
            return super().aggregate_fit(server_round, results, failures)

        # --- FEDNOVA AGGREGATION LOGIC ---
        
        # 1. Extract results and metrics
        # We expect clients to return 'tau' (number of local steps) in metrics
        grads = []
        taus = []
        num_examples_list = []
        
        for _, fit_res in results:
            # Client parameters (w_i)
            client_params = parameters_to_ndarrays(fit_res.parameters)
            
            # Calculate Pseudo-Gradient: g_i = (w_t - w_i) / tau_i
            # But wait! FedNova formula for aggregation is:
            # w_{t+1} = w_t - (Sum(p_i * tau_i * g_i) / Sum(p_i * tau_i)) * Sum(p_i * tau_i) ...
            # A simpler practical implementation is normalizing the update vector.
            
            # Get tau (local steps) from metrics. Default to 1 if missing to avoid divide-by-zero.
            tau = fit_res.metrics.get("tau", 1)
            taus.append(tau)
            
            # Calculate update vector (w_t - w_i) for this client
            # Note: We iterate through each layer (array) of the model
            grad = [
                (global_layer - client_layer) / tau 
                for global_layer, client_layer in zip(self.global_parameters_arrays, client_params)
            ]
            grads.append(grad)
            
            num_examples_list.append(fit_res.num_examples)

        # 2. Calculate Aggregation Weights (p_i)
        # p_i = n_i / n_total
        total_examples = sum(num_examples_list)
        p = [n / total_examples for n in num_examples_list]
        
        # 3. Calculate Normalized Global Update
        # Update = Sum(p_i * (w_t - w_i)) would be FedAvg.
        # FedNova: Update = Sum(p_i * (w_t - w_i) / tau_i) * (Sum(p_i * tau_i))
        # Effectively: We average the "per-step speed" of clients, then multiply by "average steps taken".
        
        # Weighted average of gradients (normalized updates)
        # weighted_grad = Sum(p_i * grad_i)
        weighted_grad = [np.zeros_like(layer) for layer in self.global_parameters_arrays]
        
        for i, grad in enumerate(grads):
            for layer_idx, layer_grad in enumerate(grad):
                weighted_grad[layer_idx] += p[i] * layer_grad

        # Calculate effective global steps (tau_eff)
        # tau_eff = Sum(p_i * tau_i)
        tau_eff = sum([p_i * tau_i for p_i, tau_i in zip(p, taus)])
        
        # 4. Apply Update to Global Model
        # w_{t+1} = w_t - tau_eff * weighted_grad
        new_global_params = [
            global_layer - (tau_eff * grad_layer)
            for global_layer, grad_layer in zip(self.global_parameters_arrays, weighted_grad)
        ]

        # 5. Return
        log(INFO, f"[FedNova] Aggregated {len(results)} clients. Avg Tau: {tau_eff:.2f}")
        return ndarrays_to_parameters(new_global_params), {}
