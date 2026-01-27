import torch
import torch.nn as nn

# --- 1. 線性回歸 (Linear) ---
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if x.dim() == 3:
            B, S, F = x.shape
            x = x.reshape(B, -1) 
        return self.linear(x)

# --- 2. 深層神經網路 (Deep MLP) ---
class DeepRegressor(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), 
            nn.ReLU(),
            nn.Dropout(0.03),  
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        if x.dim() == 3:
            B, S, F = x.shape
            x = x.reshape(B, -1)
        return self.net(x)

# --- 3. LSTM (Long Short-Term Memory) ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out