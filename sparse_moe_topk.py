# This code covers the blueprint of a Sparse Mixture of Experts (sMOE) model with top-k routing                

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
# Same as ARGMAX SMOE, like literally nothing changed. 

class SparseMixtureOfExperts(nn.Module):
    def __init__(self, embed_size, num_experts=8):
        super(SparseMixtureOfExperts, self).__init__()

        # This code is also the same as before.
        self.experts = nn.ModuleList([
            FeedForwardLayer(embed_size) 
            for _ in range(num_experts)
        ])
        self.router = Router(embed_size, num_experts)
        self.num_experts = num_experts

    def forward(self, x):

        # Run the router to get the expert probabilities
        expert_probabilities = self.router(x)

        # This code has CHANGED! We are getting the top k experts for each token.
        _, chosen_experts = torch.topk(expert_probabilities, k=2, dim=1)
        # chosen_experts Shape: [T, k] (k = 2)
        # Example output: chosen_experts = tensor([[0, 3], [1, 4], [2, 5]])
        # This means that the first token will be routed to expert 0 and expert 3
        # the second token will be routed to expert 1 and expert 4, etc.

        outputs = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):

            # Check if tokens j are assigned to this expert i
            is_token_assigned_to_expert_i = (chosen_experts == expert_idx).any(dim=-1)
            # If chosen_experts = [[0, 3], [1, 2], [0, 4], [0, 2], [3, 4], [1, 3]], and expert_idx = 1
            # Then is_token_assigned_to_expert_i = [False, True, False, False, False, True]

            # Get the indices of the tokens that are assigned to this expert
            selected_tokens = torch.nonzero(is_token_assigned_to_expert_i)[:, 0]
            # If is_token_assigned_to_expert_i = [False, True, False, False, False, True]
            # selected_tokens = [1, 5]
            # Indicating that the token 1 and 5 are assigned to expert_idx = 1
            
            if len(selected_tokens) > 0:

                # Get the tokens
                x_i = x[selected_tokens]

                # Get the ith expert
                expert_i = self.experts[expert_idx]

                # Forward pass the selected tokens through the expert
                out_i = expert_i(x_i)

                # Get the gating scores for the selected tokens
                gating_scores = expert_probabilities[selected_tokens, expert_idx]
                gating_scores = gating_scores.unsqueeze(1)

                # Scale the output of the expert by the gating scores
                out_i = out_i * gating_scores

                # Instead of saving the output, we ADD it to accumulate multiple experts' outputs.
                outputs[selected_tokens] += out_i
        
        return outputs

if __name__ == "__main__":
    smoe = SparseMixtureOfExperts(embed_size=1024, num_experts=8)
    x = torch.randn(10, 1024)
    print(smoe(x).shape)