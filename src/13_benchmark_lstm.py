# src/benchmark_lstm.py
#!/usr/bin/env python3
"""
Module 3: Benchmark Model (LSTM) - Fair Comparison Version
Purpose: Compare dCeNN-ELM against a properly sequence-aware LSTM.
Implements: Sliding window lookback, Dropout, and Weight Decay.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration for Fairness ---
LOOKBACK = 24  # LSTM now looks at the past 24 hours
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4 # Matches dCeNN config
BATCH_SIZE = 128
EPOCHS = 10 

# --- Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take only the last hidden state for the sequence
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

def create_sequences(data, target, lookback):
    """Transform 2D data into 3D sequences (N, lookback, features)."""
    X, Y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        Y.append(target[i + lookback])
    return np.array(X), np.array(Y)

def train_benchmark():
    # 1. Load Data
    data_path = "artifacts/preprocessed_datasets_dcen_elm_h1.npz"
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found. Run 'make split' first.")
        return

    data = np.load(data_path, allow_pickle=True)
    target_names = data["target_names"]
    
    # Create temporal windows
    print(f"[INFO] Creating sequences with lookback={LOOKBACK}...")
    X_train_seq, Y_train_seq = create_sequences(data["X_train"], data["Y_train"], LOOKBACK)
    X_test_seq, Y_test_seq = create_sequences(data["X_test"], data["Y_test"], LOOKBACK)

    X_train = torch.FloatTensor(X_train_seq)
    Y_train = torch.FloatTensor(Y_train_seq)
    X_test = torch.FloatTensor(X_test_seq)
    Y_test = torch.FloatTensor(Y_test_seq)

    # 2. Setup Model
    device = torch.device("cpu") # Benchmarking on CPU to simulate Edge constraints
    model = LSTMModel(in_dim=X_train.shape[2], 
                      hidden_dim=HIDDEN_DIM, 
                      out_dim=Y_train.shape[1],
                      num_layers=NUM_LAYERS,
                      dropout=DROPOUT).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)

    # 3. Benchmark Training Time
    print(f"[INFO] Starting LSTM Training Benchmark ({EPOCHS} Epochs)...")
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/len(loader):.6f}")
    
    training_duration = time.time() - start_time
    print(f"[RESULT] LSTM Training Time: {training_duration:.2f} seconds")

    # 4. Benchmark Inference Latency
    model.eval()
    latencies = []
    with torch.no_grad():
        # Warm up
        _ = model(X_test[:10])
        for i in range(min(1000, len(X_test))):
            sample = X_test[i:i+1] # Single sequential sample
            start_inf = time.perf_counter()
            _ = model(sample)
            latencies.append(time.perf_counter() - start_inf)
    
    avg_inf_latency_ms = np.mean(latencies) * 1000
    print(f"[RESULT] LSTM Avg Inference Latency: {avg_inf_latency_ms:.4f} ms per sample")

    # 5. Calculate Accuracy (RMSE)
    with torch.no_grad():
        preds = model(X_test)
    
    errors = torch.sqrt(torch.mean((preds - Y_test)**2, dim=0)).numpy()
    print("\n[RESULT] LSTM RMSE per target:")
    for name, err in zip(target_names, errors):
        print(f"  {name}: {err:.5f}")

    # 6. Save Results
    results = {
        "model": "LSTM (Baseline-Sequential)",
        "train_time_sec": training_duration,
        "inf_latency_ms": avg_inf_latency_ms,
        "rmse_solar": errors[0],
        "rmse_wind": errors[1],
        "rmse_load": errors[2],
        "rmse_price": errors[3],
        "parameters": sum(p.numel() for p in model.parameters()),
        "lookback_hours": LOOKBACK
    }
    
    os.makedirs("reports/tables", exist_ok=True)
    pd.DataFrame([results]).to_csv("reports/tables/benchmark_lstm.csv", index=False)
    print("\n[INFO] Results saved to reports/tables/benchmark_lstm.csv")

if __name__ == "__main__":
    train_benchmark()