import argparse
import subprocess
import sys
import os

def run_simulation(args):
    """
    Constructs the Flower simulation command based on arguments.
    """
    print(f"--- Starting Malware Detection FL Simulation ---")
    print(f"Model: {args.model}")
    print(f"Clients: {args.num_clients} total")
    print(f"Rounds: {args.rounds}")
    print(f"Parallelism: {args.parallel_clients} clients (Limit)")
    
    # 1. Update Resource Config dynamically (Pyproject-style override)
    # Flower 1.13+ relies heavily on pyproject.toml for 'flwr run'.
    # However, for a CLI tool, we can also use `flwr.simulation.start_simulation` in python
    # OR we can generate a temporary pyproject.toml.
    
    # For simplicity and robustness with your existing knowledge, 
    # we will use the 'flwr run' command but we need to ensure the config matches.
    # A pro tip: We can calculate resource allocation here and pass it.
    
    # Calculate CPU/GPU per client to force parallelism limit
    # Assuming 4 CPUs available (standard Kaggle/Laptop)
    total_cpus = 4.0 
    cpu_per_client = total_cpus / args.parallel_clients
    
    # Construct the command
    # We pass arguments via environment variables or writing a temporary config is best.
    # Here, we will just run the app as defined.
    
    # NOTE: To truly pass these CLI args into the 'client_fn' and 'server_fn' 
    # inside the simulation, usually requires a config file or env vars 
    # because 'flwr run' isolates the environment.
    
    os.environ["FLWR_STRATEGY_NAME"] = args.strategy  
    os.environ["FLWR_ROUNDS"] = str(args.rounds)      
    os.environ["FLWR_MODEL_NAME"] = args.model
    os.environ["DATA_DIR"] = args.data_dir
    os.environ["FLWR_FIT_CLIENTS"] = str(args.parallel_clients)    

    # We can modify the pyproject.toml on the fly or use the python API.
    # Let's use the subprocess command you are familiar with, 
    # but we will rely on the files we created in src/
    
    cmd = ["flwr", "run", ".", "local-simulation"]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower FL Malware Detection CLI")
    
    parser.add_argument("--model", type=str, default="cnn_lstm", help="Model architecture (cnn_lstm, mlp)")
    parser.add_argument("--strategy", type=str, default="fedavg", help="Aggregation strategy")
    parser.add_argument("--num_clients", type=int, default=1000, help="Total clients in pool")
    parser.add_argument("--parallel_clients", type=int, default=5, help="Max clients running in parallel")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--data_dir", type=str, default="data/", help="Path to client csv files")
    
    args = parser.parse_args()
    
    run_simulation(args)
