# This code covers the blueprint of a DenseMixtureOfExperts model.

# Dense Mixture of Experts (MoE) is a type of Mixture of Experts (MoE) 
# that executes all experts for each tokens.

import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import FeedForwardLayer

class Router(nn.Module):
    def __init__(self, embed_size, num_experts):
        super(Router, self).__init__()
        self.fc = nn.Linear(embed_size, embed_size)
        self.fc2 = nn.Linear(embed_size, num_experts)

    def forward(self, x):
        # X Shape: [T, NUM_EXPERTS]
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        # Out Shape: [T, NUM_EXPERTS], Probability distribution.
        return x
    
class DenseMixtureOfExperts(nn.Module):
    def __init__(self, embed_size, num_experts):
        super(DenseMixtureOfExperts, self).__init__()
        self.embed_size = embed_size
        self.num_experts = num_experts

        # Initialize the experts
        self.experts = nn.ModuleList(
            [
                FeedForwardLayer(embed_size) 
                for _ in range(num_experts)
            ]
        )

        # Initialize the router layer
        self.router = Router(embed_size=embed_size, num_experts=num_experts)

    def forward(self, x):

        # X Shape: [T, D]
        expert_probabilities = self.router(x)
        # Expert Probabilities Shape: [T, NUM_EXPERTS]

        # Pass each token through all experts and stack the results.
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) 
        # Expert Outputs Shape: [T, NUM_EXPERTS, D]

        # Scale the expert outputs by the expert probabilities.
        expert_outputs = expert_outputs * expert_probabilities.unsqueeze(-1)
        # Scaled outputs Shape: [T, NUM_EXPERTS, D]

        # Sum the expert outputs over the NUM_EXPERTS dimension.
        expert_outputs = expert_outputs.sum(dim=1)
        # Out Shape: [T, D]
        
        return expert_outputs
        

if __name__ == "__main__":
    dense_moe = DenseMixtureOfExperts(embed_size=1024, num_experts=8)
    x = torch.randn(10, 1024)
    print(dense_moe(x).shape)