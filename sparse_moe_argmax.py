import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import FeedForwardLayer

# The Router inputs token embeddings and outputs a probability of each token to each expert.
class Router(nn.Module):
    def __init__(self, embed_size, num_experts):
        super(Router, self).__init__()
        self.fc = nn.Linear(embed_size, embed_size)
        self.fc2 = nn.Linear(embed_size, num_experts)
    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x
# Same as before, like literally nothing changed. 

class SparseMixtureOfExperts(nn.Module):
    def __init__(self, embed_size, num_experts=8):
        super(SparseMixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([
            FeedForwardLayer(embed_size) 
            for _ in range(num_experts)
        ])
        self.router = Router(embed_size, num_experts)

    def forward(self, x):
        # Get the expert probabilities for each token
        expert_probabilities = self.router(x)

        # Choose the highest probability expert for each token. Input: [T, NUM_EXPERTS]
        chosen_experts = torch.argmax(expert_probabilities, dim=1)
        # Example output: chosen_experts = tensor([0, 3, 1, 4])
        # This means that the first token will be routed to expert 0 
        # the second token will be routed to expert 3, etc.

        outputs = torch.zeros_like(x)

        for i in range(len(chosen_experts)):

            # Find the indices of the tokens that are routed to expert i
            selected_tokens = torch.nonzero(chosen_experts == i)[:, 0]

            if len(selected_tokens) > 0:

                # Get the tokens that are routed to expert i
                x_i = x[selected_tokens]
                # x_i Shape: [num_selected_tokens, embed_size], num_selected_tokens <= T

                # Get the output of expert i
                expert_i = self.experts[i]

                # Forward pass the tokens through the expert
                out_i = expert_i(x_i)

                # Save the output for each token in the final output tensor
                outputs[selected_tokens] = out_i
        
        return outputs


if __name__ == "__main__":
    smoe = SparseMixtureOfExperts(embed_size=1024, num_experts=8)
    x = torch.randn(10, 1024)
    print(smoe(x).shape)