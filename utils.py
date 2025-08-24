import torch.nn as nn
import torch.nn.functional as F

class FeedForwardLayer(nn.Module):
    def __init__(self, embed_size):
        super(FeedForwardLayer, self).__init__()
        self.embed_size = embed_size
        self.fc1 = nn.Linear(embed_size, embed_size)
        self.fc2 = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # X Shape: [T, D]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # Out Shape: [T, D]
        return x