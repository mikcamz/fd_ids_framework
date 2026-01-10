# Federated Malware Detection Tool 🛡️

A modular, command-line based **Federated Learning (FL) simulation framework** designed for network intrusion detection. Built with **Flower (Flwr)**, **PyTorch**, and **Ray**, this tool simulates decentralized training on the **CIC-IOT 2023** dataset with support for Non-IID data distributions and advanced aggregation strategies.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flower](https://img.shields.io/badge/Flower-1.13%2B-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

## 🌟 Key Features

* **Scalable Simulation:** Runs hundreds of clients efficiently using Ray backend with dynamic CPU/GPU resource allocation.
* **Modular Architecture:** Easily swap models (`CNN-LSTM`, `MLP`) and strategies (`FedAvg`, `FedProx`, `FedNova`) via configuration.
* **Non-IID Data Generation:** Includes tools to simulate real-world data skew (feature and quantity skew) across clients.
* **Advanced Strategies:**
    * **FedNova:** Normalized averaging for clients with varying local steps.
    * **Dynamic Scaling:** Custom strategy to scale from low (3 clients) to high (5+ clients) parallelism mid-simulation.
* **Production Ready:** Uses `src/` layout, `argparse` CLI, and standard `pyproject.toml` configuration.

## 📂 Project Structure

```text
malware-detection-tools/
├── data/                    # Holds generated client CSVs (Ignored by Git)
├── outputs/                 # Simulation logs and results
├── tools/
│   └── generate_data.py     # Script to split raw dataset into Non-IID partitions
├── src/
│   ├── client.py            # Flower Client Logic
│   ├── server.py            # Server & Aggregation Logic
│   ├── dataset.py           # Data Loading & Preprocessing
│   ├── models/              # Neural Network Architectures
│   │   └── cnn_lstm.py      # CNN-LSTM Implementation
│   └── strategies/          # Custom FL Strategies
│       ├── fednova.py       # FedNova Implementation
│       └── dynamic.py       # Dynamic Resource Scaling
├── main.py                  # CLI Entry Point
└── pyproject.toml           # Flower Simulation Config
```

## Installation

   1. Clone the repository:
```bash
git clone [https://github.com/your-username/malware-detection-tools.git](https://github.com/your-username/malware-detection-tools.git)
cd malware-detection-tools
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Note: Requires flwr[simulation], torch, pandas, numpy, scikit-learn.

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

Run the simulation using main.py. You can configure the model, strategy, and parallelism via command-line arguments.
1. Basic Run (FedAvg)
```Bash
python main.py --model cnn_lstm --strategy fedavg --parallel_clients 5 --rounds 10
```

2. Using FedNova (Robust Aggregation)
Ideal for scenarios where clients perform different amounts of work.
```Bash
python main.py --model cnn_lstm --strategy fednova --parallel_clients 5 --rounds 20
```

3. Dynamic Parallelism (Custom Strategy)
Starts with 3 clients to warm up, then scales to 5 clients after round 3.
```Bash
python main.py --strategy dynamic --rounds 10
```

## Customization

### Adding a New Model
- Create `src/models/my_model.py` defining your PyTorch class.
- Register it in `src/models/__init__.py`.
- Run with `--model my_model`.

### Adding a New Strategy
- Create `src/strategies/my_strategy.py`.
- Register it in `src/strategies/__init__.py`.
- Run with `--strategy my_strategy`.

## Citation

If you use this tool for research, please credit the original CIC-IOT dataset providers and the Flower framework.
