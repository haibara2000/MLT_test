import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim] -> [batch_size, hidden_dim]
        _, (h_n, _) = self.lstm(x)  # 只取最后一个时间步的隐状态
        x = h_n[-1]  # [batch_size, hidden_dim]
        x = self.fc(x)  # [batch_size, output_dim]
        return x
