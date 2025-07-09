# adapter_gpt.py
import torch
from torch import nn

class VQEncoder(nn.Module):
    def __init__(self, vocab_size, code_dim=512, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, code_dim)
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return (torch.sigmoid(self.fc2(h)) > 0.5).float()

class VQDecoder(nn.Module):
    def __init__(self, vocab_size, code_dim=512, hidden=512):
        super().__init__()
        self.fc3 = nn.Linear(code_dim, hidden)
        self.fc4 = nn.Linear(hidden, vocab_size)
    def forward(self, c):
        h = torch.relu(self.fc3(c))
        return self.fc4(h)

class Denoiser(nn.Module):
    def __init__(self, code_dim=512, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, code_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)
