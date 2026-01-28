import os
import glob
import pandas as pd
import numpy as np
import argparse

def generate_partitions(source_dir, output_dir, num_clients):
    print(f"--- Step 1: Loading Global Data Pool from {source_dir} ---")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    all_files = glob.glob(os.path.join(source_dir, "*.csv"))
    all_files.sort()

    if not all_files:
        print(f"[Warning] No CSV files found in {source_dir}. Aborting")
    else:
        print(f"Found {len(all_files)} source files. Loading first 15 (RAM limit)...")
        df_list = []
        # Limit to first 15 files to prevent OOM, just like your notebook
        for f in all_files[:15]: 
            try:
                df = pd.read_csv(f)
                df_list.append(df)
            except Exception as e: 
                print(f"Skipping broken file {f}: {e}")
        
        if not df_list:
            print("Error: Could not read any files.")
            return

        df_global = pd.concat(df_list, ignore_index=True)

    # Clean unrelated columns
    if 'Protocol Type' in df_global.columns:
        df_global.drop('Protocol Type', axis=1, inplace=True)
        
    df_global.reset_index(drop=True, inplace=True)
    print(f"Global Pool Loaded: {df_global.shape[0]} samples.")

    # ==========================================
    # 2. FAST INDEXING
    # ==========================================
    print("\n--- Step 2: Indexing Data (Speed Optimization) ---")
    # This map allows instant lookup of all row indices for a specific label
    label_indices_map = df_global.groupby('Label').indices 
    all_unique_labels = list(label_indices_map.keys())
    print(f"Found {len(all_unique_labels)} unique labels.")

    # ==========================================
    # 3. GENERATE SEPARATE CSVs
    # ==========================================
    print(f"\n--- Step 3: Generating {num_clients} Separate CSV Files ---")
    summary_stats = {}

    for client_id in range(num_clients):
        if client_id % 10 == 0: 
            print(f"Processing Client {client_id}/{num_clients}...", end="\r")
        
        # --- YOUR LOGIC START ---
        
        # RULE 1: Random Labels (Select between 10 and 34 classes)
        # This creates "Feature/Label Skew" (Non-IID)
        n_labels_to_pick = np.random.randint(10, 35)
        n_labels_to_pick = min(n_labels_to_pick, len(all_unique_labels))
        
        selected_labels = np.random.choice(all_unique_labels, n_labels_to_pick, replace=False)
        
        client_indices = []
        
        # Collect row numbers for this client
        for label in selected_labels:
            available_indices = label_indices_map[label]
            total_samples_for_label = len(available_indices)
            
            if total_samples_for_label == 0: continue

            # RULE 2: Random Samples (Take 0.1% to 1.0% of available data for this label)
            # This creates "Quantity Skew"
            ratio = np.random.uniform(0.001, 0.01)
            n_samples = int(total_samples_for_label * ratio)
            n_samples = max(1, n_samples) # Ensure at least 1 sample if selected
            
            selected_rows = np.random.choice(available_indices, n_samples, replace=False)
            client_indices.extend(selected_rows)
            
            # Record for summary
            if label not in summary_stats: summary_stats[label] = {}
            summary_stats[label][f"C{client_id}"] = n_samples

        # --- SAVE THE SEPARATE FILE ---
        if client_indices:
            client_df = df_global.loc[client_indices]
            save_path = os.path.join(output_dir, f"client_{client_id}.csv")
            client_df.to_csv(save_path, index=False)
        else:
            print(f"Warning: Client {client_id} ended up with 0 samples.")
        
        # --- YOUR LOGIC END ---

    print(f"\nDone! {num_clients} separate CSV files created in {output_dir}")

    # Save Summary Matrix (Useful for debugging distribution)
    summary_path = os.path.join(output_dir, "summary_table.csv")
    pd.DataFrame.from_dict(summary_stats).fillna(0).T.to_csv(summary_path)
    print(f"Distribution summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Non-IID Client Data from CIC-IOT")
    
    # Default points to where you might mount the Kaggle dataset on your local PC
    # Or just use a folder where you put the big CSVs
    parser.add_argument("-s", "--source", type=str, default="./raw_data", help="Folder containing raw merged CSVs")
    parser.add_argument("-o", "--output", type=str, default="./data", help="Output folder for client_x.csv")
    parser.add_argument("-n", "--num_clients", type=int, default=100, help="Number of clients to simulate")
    
    args = parser.parse_args()
    
    generate_partitions(args.source, args.output, args.num_clients)
