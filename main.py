import argparse
import os
import sys
import yaml
import json
import subprocess

def load_config(path="config.yaml"):
    if not os.path.exists(path):
        print(f"Config file {path} not found. Using defaults.")
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_simulation():
    # 1. Load Config
    config = load_config("config.yaml")
    sim_config = config.get('simulation', {})
    res_config = config.get('resources', {})
    schedule_config = config.get('dynamic_schedule', [])
    
    # 2. CLI Args
    parser = argparse.ArgumentParser(description="Run FL Simulation")
    parser.add_argument("-r", "--rounds", type=int, default=None, help="Number of FL rounds")
    parser.add_argument("-s", "--strategy", type=str, default=None, help="FL Strategy (fedavg, fednova, etc.)")
    parser.add_argument("-m", "--model", type=str, default=None, help="Model architecture (cnn_lstm, etc.)")
    parser.add_argument("-n", "--num_clients", type=int, default=None, help="Total number of clients in the pool")
    parser.add_argument("-c", "--clients_overwrite", type=int, default=None, help="Force specific number of active clients")
    parser.add_argument("-d", "--data_dir", type=str, default=None, help="Path to client datasets")
    args, unknown = parser.parse_known_args() # Use parse_known_args to be safe

    # 3. Resolve Values
    num_rounds = args.rounds if args.rounds else sim_config.get('rounds', 5)
    
    # Resolve Total Clients (CLI > Config > Default)
    num_clients = args.num_clients if args.num_clients else sim_config.get('num_clients', 1000)
    model_name = args.model if args.model else sim_config.get('model', 'cnn_lstm')
    gpu_per_client = res_config.get('gpu_per_client', 0.0)
    cpu_per_client = res_config.get('cpu_per_client', 1.0)
    data_dir = args.data_dir if args.data_dir else sim_config.get('data_dir', 'data/')
    strategy_name = args.strategy if args.strategy else sim_config.get('strategy', 'fedavg')
    if args.clients_overwrite:
        schedule = [{"round_start": 1, "active_clients": args.clients_overwrite}]
    else:
        schedule = schedule_config

    env = os.environ.copy()
    env["FLWR_MODEL_NAME"] = model_name
    env["FLWR_STRATEGY_NAME"] = strategy_name
    env["FLWR_ROUNDS"] = str(num_rounds)
    env["DATA_DIR"] = data_dir
    env["DYNAMIC_SCHEDULE"] = json.dumps(schedule)

    fed_config_str = (
        f"options.num-supernodes={num_clients} "
        f"options.backend.client-resources.num-cpus={cpu_per_client} "
        f"options.backend.client-resources.num-gpus={gpu_per_client}"
    )

    # 4. Build Command
    cmd = [
        "flwr", "run", ".", "local-simulation",
        
        # Override Resources
        "--federation-config", fed_config_str,
    ]

    print(f"--- Configuration ---")
    print(f"Total Clients: {num_clients}")
    print(f"Executing: {' '.join(cmd)}")
    
    # ... (subprocess execution) ...
    python_bin_dir = os.path.dirname(sys.executable)
    env["PATH"] = python_bin_dir + os.pathsep + env.get("PATH", "")
    subprocess.run(cmd, env=env, check=True)

if __name__ == "__main__":
    run_simulation()