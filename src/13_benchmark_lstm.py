import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
LOOKBACK = 24
HIDDEN_DIM = 64
EPOCHS = 30 # Increased to ensure it moves away from 0
LEARNING_RATE = 0.001

class LSTMModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_benchmark():
    data = np.load("artifacts/preprocessed_datasets_dcen_elm_h1.npz", allow_pickle=True)
    X_tr, Y_tr = data["X_train"], data["Y_train"]
    X_te, Y_te = data["X_test"], data["Y_test"]

    # CRITICAL: Scale targets for the LSTM
    scaler_y = StandardScaler().fit(Y_tr)
    Y_tr_s = scaler_y.transform(Y_tr)
    Y_te_s = scaler_y.transform(Y_te)

    # Sequence Building
    X_seq = np.array([X_tr[i:i+LOOKBACK] for i in range(len(X_tr)-LOOKBACK)])
    Y_seq = Y_tr_s[LOOKBACK:]

    model = LSTMModel(X_tr.shape[1], HIDDEN_DIM, Y_tr.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_seq), torch.FloatTensor(Y_seq)), batch_size=128, shuffle=True)

    model.train()
    for ep in range(EPOCHS):
        for bx, by in loader:
            optimizer.zero_grad(); loss = nn.MSELoss()(model(bx), by); loss.backward(); optimizer.step()

    # SAVE EVERYTHING
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/lstm_model.pt")
    np.savez("artifacts/scaler_y.npz", mean=scaler_y.mean_, scale=scaler_y.scale_)
    print("[INFO] LSTM Trained and Artifacts Saved.")

if __name__ == "__main__":
    train_benchmark()