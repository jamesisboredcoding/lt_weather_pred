import torch.nn as nn

class LTSM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=256, num_layers=2, dropout=.2):
        super().__init__()
        self.ltsm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim / 2), output_size)
        )

    def forward(self, x):
        out, (h_n, c_n) = self.ltsm(x)
        last_hidden = h_n[-1]
        return self.head(last_hidden).squeeze(-1)