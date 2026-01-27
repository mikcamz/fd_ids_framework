# Federated Malware Detection Tool 🛡️

A modular, command-line based **Federated Learning (FL) simulation framework** designed for network intrusion detection. Built with **Flower (Flwr)**, **PyTorch**, and **Ray**, this tool simulates decentralized training on the **CIC-IOT 2023** dataset with support for Non-IID data distributions and advanced aggregation strategies.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flower](https://img.shields.io/badge/Flower-1.13%2B-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

## 🌟 Key Features

* **Scalable Simulation:** Runs hundreds of clients efficiently using Ray backend with dynamic CPU/GPU resource allocation.
* **YAML-Based Configuration:** Centralized control over model, strategy, and resources via `config.yaml`.
* **Modular Architecture:** Easily swap models (`CNN-LSTM`, `MLP`) and strategies (`FedAvg`, `FedProx`, `FedNova`).
* **Non-IID Data Generation:** Includes tools to simulate real-world data skew (feature and quantity skew) across clients.
* **Custom Dynamic Schedules:** Define custom scaling logic in YAML (e.g., "Start with 2 clients, scale to 10 after round 5") to simulate network variability or warm-up phases.
* **Production Ready:** Uses `src/` layout, `argparse` CLI with short-flags, and standard `pyproject.toml` configuration.

## 📂 Project Structure

```text
malware-detection-tools/
├── config.yaml              # Main simulation configuration & dynamic schedule
├── data/                    # Holds generated client CSVs (Ignored by Git)
├── outputs/                 # Simulation logs and results
├── tools/
│   └── generate_data.py     # Script to split raw dataset into Non-IID partitions
├── src/
│   ├── client.py            # Flower Client Logic
│   ├── server.py            # Server & Aggregation Logic
│   ├── dataset.py           # Data Loading & Preprocessing
│   ├── models/              # Neural Network Architectures
│   └── strategies/          # Custom FL Strategies (FedNova, Dynamic)
├── main.py                  # CLI Entry Point
└── pyproject.toml           # Flower Simulation Config
```

## Installation

   1. Clone the repository:
```bash
git clone https://github.com/mikcamz/fd_ids_framework.git
cd malware-detection-tools
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Note: Requires flwr[simulation], torch, pandas, numpy, scikit-learn, pyyaml.

## Data Preparation

Before running the simulation, you must generate the client datasets. This script mimics Non-IID distribution by assigning random classes and sample sizes to each client.

- Download the CIC-IOT 2023 dataset (CSVs).
- Run the generator:
```Bash
# Example: Generate 100 clients from raw data
python tools/generate_data.py --source ./path_to_raw_csvs --output ./data --num_clients 100
```
This creates files data/client_0.csv to data/client_99.csv.

## Usage

1. Run using Config File (Recommended)

Edit config.yaml to set your desired rounds, strategy, and schedule, then simply run:
```bash
python main.py
```
2. CLI Overrides

You can override specific parameters without changing the config file:
Example: Quick Test Run 5 rounds with FedNova, using a total pool of 50 clients:
```bash
python main.py -r 5 -s fednova -n 50 -d data/
```
3. Edit custom parallel scheduling in `config.yaml`:
```yaml
resources:
  total_gpus: 1.0
  total_cpus: 4.0
  gpu_per_client: 0.0
  cpu_per_client: 1.0

dynamic_schedule:
  - round_start: 1       # At round 1, 2 client will run in parallel
    active_clients: 2
  - round_start: 4       # Starting from round 4, 5 client will run in parallel
    active_clients: 5
```

## Customization

### Adding a New Model
- Create `src/models/my_model.py` defining your PyTorch class.
- Register it in `src/models/__init__.py`.
- Run with `-m my_model`.

### Adding a New Strategy
- Create `src/strategies/my_strategy.py`.
- Register it in `src/strategies/__init__.py`.
- Run with `-s my_strategy`.

## Citation

If you use this tool for research, please credit the original CIC-IOT dataset providers and the Flower framework.
