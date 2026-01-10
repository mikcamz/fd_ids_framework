import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Global constants for this dataset (34 Classes)
ALL_CLASSES = [
    'BACKDOOR_MALWARE', 'BENIGN', 'BROWSERHIJACKING', 'COMMANDINJECTION', 
    'DDOS-ACK_FRAGMENTATION', 'DDOS-HTTP_FLOOD', 'DDOS-ICMP_FLOOD', 'DDOS-ICMP_FRAGMENTATION', 
    'DDOS-PSHACK_FLOOD', 'DDOS-RSTFINFLOOD', 'DDOS-SLOWLORIS', 'DDOS-SYNONYMOUSIP_FLOOD', 
    'DDOS-SYN_FLOOD', 'DDOS-TCP_FLOOD', 'DDOS-UDP_FLOOD', 'DDOS-UDP_FRAGMENTATION', 
    'DICTIONARYBRUTEFORCE', 'DNS_SPOOFING', 'DOS-HTTP_FLOOD', 'DOS-SYN_FLOOD', 
    'DOS-TCP_FLOOD', 'DOS-UDP_FLOOD', 'MIRAI-GREETH_FLOOD', 'MIRAI-GREIP_FLOOD', 
    'MIRAI-UDPPLAIN', 'MITM-ARPSPOOFING', 'RECON-HOSTDISCOVERY', 'RECON-OSSCAN', 
    'RECON-PINGSWEEP', 'RECON-PORTSCAN', 'SQLINJECTION', 'UPLOADING_ATTACK', 
    'VULNERABILITYSCAN', 'XSS'
]

def load_client_data(partition_id: int, data_dir: str, batch_size: int, n_components: int = 20):
    """
    Loads data for a specific partition_id.
    
    Args:
        partition_id: The ID of the client (0, 1, 2...)
        data_dir: Directory containing 'client_0.csv', 'client_1.csv', etc.
        batch_size: Batch size for DataLoader
        n_components: Number of PCA components to keep
    """
    # Find all generated files to determine modulo
    # (e.g., if we have 100 files but run 1000 clients, client 100 gets file 0)
    generated_files = glob.glob(os.path.join(data_dir, "client_*.csv"))
    num_files = len(generated_files)
    
    # --- ERROR HANDLING: NO FILES ---
    if num_files == 0:
        print(f"[Warning] No client CSVs found in {data_dir}. Using DUMMY data.")
        # Return Dummy Data so simulation doesn't crash
        X = np.random.rand(10, n_components).astype(np.float32)
        y = np.random.randint(0, len(ALL_CLASSES), 10).astype(np.int64)
        X = X.reshape((10, 1, n_components))
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size), DataLoader(ds, batch_size=batch_size)

    # Calculate specific file for this client
    file_idx = partition_id % num_files
    target_file = os.path.join(data_dir, f"client_{file_idx}.csv")
    
    try:
        df = pd.read_csv(target_file)
        
        # --- CRITICAL FIX 1: CLEAN DATA ---
        # Remove Infinity values which crash PyTorch Training
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # --- CRITICAL FIX 2: CLEAN LABELS ---
        # Ensure labels match ALL_CLASSES exactly (trim spaces, uppercase)
        if 'Label' in df.columns:
            df['Label'] = df['Label'].astype(str).str.strip().str.upper()
            
        # Encode Labels
        le = LabelEncoder()
        le.fit(ALL_CLASSES)
        df = df[df['Label'].isin(ALL_CLASSES)]
        df['Label'] = le.transform(df['Label'])
        
        X_raw = df.drop(['Label'], axis=1, errors='ignore').values
        y_raw = df['Label'].values
        
        # Scaling & PCA
        scaler = StandardScaler()
        if len(X_raw) > 1:
            X_scaled = scaler.fit_transform(X_raw)
            actual_components = min(n_components, len(X_raw))
            pca = PCA(n_components=actual_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Pad with zeros if we didn't get enough components (rare edge case)
            if actual_components < n_components:
                padding = np.zeros((len(X_raw), n_components - actual_components))
                X_pca = np.hstack([X_pca, padding])
        else:
            # Not enough data for PCA
            X_pca = np.zeros((len(X_raw), n_components))
            
        # Reshape for CNN/LSTM (Batch, Channels, Features)
        X = X_pca.reshape((X_pca.shape[0], 1, n_components)).astype(np.float32)
        y = y_raw.astype(np.int64)
        
        # Split Train/Test (80/20)
        split_idx = int(0.8 * len(X))
        if split_idx == 0 and len(X) > 0: split_idx = 1
        
        train_ds = TensorDataset(torch.from_numpy(X[:split_idx]), torch.from_numpy(y[:split_idx]))
        test_ds = TensorDataset(torch.from_numpy(X[split_idx:]), torch.from_numpy(y[split_idx:]))
        
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size)
        
    except Exception as e:
        print(f"Error loading {target_file}: {e}")
        # Fallback to dummy data on error
        X = np.random.rand(10, n_components).astype(np.float32)
        y = np.random.randint(0, len(ALL_CLASSES), 10).astype(np.int64)
        X = X.reshape((10, 1, n_components))
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size), DataLoader(ds, batch_size=batch_size)
