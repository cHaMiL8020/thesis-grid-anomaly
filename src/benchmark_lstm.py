# src/benchmark_lstm.py
#!/usr/bin/env python3
"""
Module 3: Benchmark Model (LSTM)
Purpose: Compare the dCeNN-ELM approach against a standard LSTM.
Uses the same dataset from artifacts/preprocessed_datasets_dcen_elm_h1.npz.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# --- Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        # LSTM layer to capture temporal dependencies
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected layer to map hidden state to target outputs
        self.fc = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        # x shape: (batch, features) -> transform to (batch, seq_len=1, features)
        x = x.unsqueeze(1) 
        out, _ = self.lstm(x)
        # Use the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def train_benchmark():
    # 1. Load Data
    data_path = "artifacts/preprocessed_datasets_dcen_elm_h1.npz"
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at {data_path}. Run 'make split' first.")
        return

    data = np.load(data_path, allow_pickle=True)
    X_train = torch.FloatTensor(data["X_train"])
    Y_train = torch.FloatTensor(data["Y_train"])
    X_test = torch.FloatTensor(data["X_test"])
    Y_test = torch.FloatTensor(data["Y_test"])
    target_names = data["target_names"]

    # 2. Setup Model
    device = torch.device("cpu") # Benchmarking on CPU to simulate Edge constraints
    model = LSTMModel(in_dim=X_train.shape[1], hidden_dim=64, out_dim=Y_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=128, shuffle=True)

    # 3. Benchmark Training Time
    print("[INFO] Starting LSTM Training Benchmark (5 Epochs)...")
    start_time = time.time()
    for epoch in range(5): 
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"  Epoch {epoch+1}/5 - Loss: {epoch_loss/len(loader):.6f}")
    
    training_duration = time.time() - start_time
    print(f"[RESULT] LSTM Training Time: {training_duration:.2f} seconds")

    # 4. Benchmark Inference Latency
    # We measure this per-sample to simulate real-time edge performance
    model.eval()
    latencies = []
    with torch.no_grad():
        # Warm up
        _ = model(X_test[:10])
        
        for i in range(min(1000, len(X_test))):
            sample = X_test[i:i+1]
            start_inf = time.perf_counter()
            _ = model(sample)
            latencies.append(time.perf_counter() - start_inf)
    
    avg_inf_latency_ms = np.mean(latencies) * 1000
    print(f"[RESULT] LSTM Avg Inference Latency: {avg_inf_latency_ms:.4f} ms per sample")

    # 5. Calculate Accuracy (RMSE)
    with torch.no_grad():
        preds = model(X_test)
    
    # Calculate RMSE for each target (e.g., Solar, Wind, Load, Price)
    errors = torch.sqrt(torch.mean((preds - Y_test)**2, dim=0)).numpy()
    
    print("\n[RESULT] LSTM RMSE per target:")
    for name, err in zip(target_names, errors):
        print(f"  {name}: {err:.5f}")

    # 6. Save results for the Master Table
    results = {
        "model": "LSTM (Baseline)",
        "train_time_sec": training_duration,
        "inf_latency_ms": avg_inf_latency_ms,
        "rmse_solar": errors[0],
        "rmse_wind": errors[1],
        "rmse_load": errors[2],
        "rmse_price": errors[3],
        "parameters": sum(p.numel() for p in model.parameters())
    }
    
    # Create tables dir if missing
    os.makedirs("reports/tables", exist_ok=True)
    pd.DataFrame([results]).to_csv("reports/tables/benchmark_lstm.csv", index=False)
    print("\n[INFO] Individual LSTM results saved to reports/tables/benchmark_lstm.csv")

if __name__ == "__main__":
    train_benchmark()