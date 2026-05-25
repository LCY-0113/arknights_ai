# training/train_tactical.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json

# ---------- 数据预处理 ----------
class ArknightsDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        # 构建动作字典
        self.action2idx = {...}  # 将动作映射为整数

    def __len__(self):
        return len(self.data) - 5   # 需要 5 帧序列

    def __getitem__(self, idx):
        # 取连续 5 帧状态
        states = []
        for i in range(idx, idx+5):
            state = self.data[i][0]
            states.append(self._state_to_tensor(state))
        states = torch.stack(states)          # (5, state_dim)
        action = self.action2idx[self.data[idx+4][1]]
        return states, torch.tensor(action)

    def _state_to_tensor(self, state):
        # 特征工程：将 JSON 状态转为固定长度向量
        # 你可以用简单的编码（one-hot + 数值归一化）
        pass

# ---------- 模型定义 ----------
class TacticalTransformer(nn.Module):
    def __init__(self, state_dim=128, num_actions=50, seq_len=5):
        super().__init__()
        self.encoder = nn.Linear(state_dim, 256)
        transformer_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        self.fc = nn.Linear(256, num_actions)

    def forward(self, x):
        # x: (batch, seq_len, state_dim)
        x = torch.relu(self.encoder(x))
        x = x.permute(1, 0, 2)   # (seq_len, batch, dim)
        x = self.transformer(x)
        x = x[-1]                # 取最后一帧的输出
        return self.fc(x)

# ---------- 训练循环 ----------
def train():
    dataset = ArknightsDataset("expert_trajectories.json")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = TacticalTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        for states, actions in loader:
            logits = model(states)
            loss = criterion(logits, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "tactical_model.pth")