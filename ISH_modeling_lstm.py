import torch
import numpy as np
import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

def create_sequences(df, assets, window=60):
    X, y = [], []
    arr = df.values
    for i in range(window, len(df)):
        X.append(arr[i-window:i])
        y.append(arr[i, [df.columns.get_loc(f'{a}_ret') for a in assets]])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, 'data', 'processed')
assets = ["S&P500", "FTSE100", "Nikkei225", "EEM", "Gold", "UST10Y"]
train_df = pd.read_csv(os.path.join(input_dir, 'train_norm.csv'), index_col=0)
test_df = pd.read_csv(os.path.join(input_dir, 'test_norm.csv'), index_col=0)

X_train, y_train = create_sequences(train_df, assets, window=60)
X_test, y_test = create_sequences(test_df, assets, window=60)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)

model = LSTMModel(input_dim=X_train.shape[2], hidden_dim=50, output_dim=len(assets)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

for epoch in range(20):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        yhat = model(xb)
        loss = criterion(yhat, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.shape[0]
    print(f"Epoch {epoch+1}: train loss {total_loss / len(X_train):.6f}")

torch.save(model.state_dict(), os.path.join(base_dir, "lstm_trained.pth"))

model.eval()
with torch.no_grad():
    preds = model(X_test).cpu().numpy()
# CLIP predictions for robustness and save
preds = np.clip(preds, -0.01, 0.01)
dates = test_df.index[60:]
pred_df = pd.DataFrame(preds, columns=assets, index=dates)
pred_df.to_csv(os.path.join(base_dir, 'pred_returns_lstm.csv'))
print("LSTM test predictions saved (clipped for robustness).")
