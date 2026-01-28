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

def get_flwr_executable():
    """
    Finds the 'flwr' executable relative to the current Python interpreter.
    This ensures we use the version installed in the active virtual environment.
    """
    # Get the directory where the current python is running (e.g., venv/bin/)
    python_dir = os.path.dirname(sys.executable)
    
    # Check for Windows (.exe) or Linux/Mac (no extension)
    if os.name == 'nt':
        candidates = ["flwr.exe", "flwr"]
    else:
        candidates = ["flwr"]
        
    for candidate in candidates:
        flwr_path = os.path.join(python_dir, candidate)
        if os.path.exists(flwr_path):
            return flwr_path
            
    # Fallback: If not found in the bin folder, hope it's in the system PATH
    return "flwr"

def run_simulation():
    # 1. Load Config
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0" # Disable boring log
    os.environ["RAY_memory_monitor_refresh_ms"] = "100" # Check memory often
    os.environ["RAY_memory_usage_threshold"] = "0.85"   # Kill workers if RAM > 85%
    config = load_config("config.yaml")
    sim_config = config.get('simulation', {})
    res_config = config.get('resources', {})
    
    # 2. CLI Args
    parser = argparse.ArgumentParser(description="Run FL Simulation")
    parser.add_argument("-r", "--rounds", type=int, default=None)
    parser.add_argument("-s", "--strategy", type=str, default=None)
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("-n", "--num_clients", type=int, default=None)
    parser.add_argument("-c", "--clients_overwrite", type=int, default=None)
    parser.add_argument("-d", "--data_dir", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default=None, help="Save all output to this file (e.g., log.txt)")
    
    args, unknown = parser.parse_known_args()

    # 3. Resolve Values
    num_rounds = args.rounds if args.rounds else sim_config.get('rounds', 5)
    strategy_name = args.strategy if args.strategy else sim_config.get('strategy', 'fedavg')
    model_name = args.model if args.model else sim_config.get('model', 'cnn_lstm')
    data_dir = args.data_dir if args.data_dir else sim_config.get('data_dir', 'data/')
    num_clients = args.num_clients if args.num_clients else sim_config.get('num_clients', 1000)

    # Schedule Logic
    schedule_config = config.get('dynamic_schedule', [])
    if args.clients_overwrite:
        schedule = [{"round_start": 1, "active_clients": args.clients_overwrite}]
    else:
        schedule = schedule_config

    # Resources
    gpu_per_client = res_config.get('gpu_per_client', 0.0)
    cpu_per_client = res_config.get('cpu_per_client', 1.0)
    
    # 4. Set Environment Variables
    env = os.environ.copy()
    env["FLWR_MODEL_NAME"] = model_name
    env["FLWR_STRATEGY_NAME"] = strategy_name
    env["FLWR_ROUNDS"] = str(num_rounds)
    env["DATA_DIR"] = data_dir
    env["DYNAMIC_SCHEDULE"] = json.dumps(schedule)

    # 5. Build Command
    # Use the dynamic executable finder
    flwr_cmd = get_flwr_executable()

    fed_config_str = (
        f"options.num-supernodes={num_clients} "
        f"options.backend.client-resources.num-cpus={cpu_per_client} "
        f"options.backend.client-resources.num-gpus={gpu_per_client}"
    )

    cmd = [
        flwr_cmd, "run", ".", "local-simulation",
        "--federation-config", fed_config_str,
    ]

    print(f"--- Configuration ---")
    print(f"Executable: {flwr_cmd}")
    print(f"Model: {model_name} | Strategy: {strategy_name}")
    print(f"World Size: {num_clients} Clients")
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        if args.output:
            # "Tee" Mode: Write to file AND console
            with open(args.output, "w") as log_file:
                # Merge stdout and stderr so everything goes to the log
                process = subprocess.Popen(
                    cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, encoding='utf-8'
                )

                # Real-time streaming
                for line in process.stdout:
                    sys.stdout.write(line) # Print to screen
                    log_file.write(line)   # Write to file


                process.wait()
                if process.returncode != 0:
                    print(f"\nSimulation failed with exit code {process.returncode}")
        else:
            # Normal Mode (Just run)
            subprocess.run(cmd, env=env, check=True)
    except FileNotFoundError:
        print(f"\nCRITICAL ERROR: Could not find '{flwr_cmd}'.")
        print(f"Ensure 'flwr' is installed in your current environment.")
    except Exception as e:
        print(f"\nExecution Error: {e}")


if __name__ == "__main__":
    run_simulation()