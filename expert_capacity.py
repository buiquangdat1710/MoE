import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import FeedForwardLayer


# Same as before
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

def randomly_select(tokens, capacity):
    return tokens[torch.randperm(len(tokens))[:capacity]]

class BasicSwitchTransformer(nn.Module):
    def __init__(self, embed_size, num_experts=8, capacity_factor=1):
        super(BasicSwitchTransformer, self).__init__()

        self.capacity_factor = capacity_factor
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            FeedForwardLayer(embed_size) 
            for _ in range(num_experts)
        ])
        self.router = Router(embed_size, num_experts)

    def forward(self, x):
        T, D = x.shape # 10, 1024

        # Calculate the capacity of each expert
        expert_capacity = int(self.capacity_factor * T / self.num_experts) # expert_capacity =  (1 * 10 / 8) = 1

        # Higher the capacity factor, the more tokens each expert will process.
        
        # Run the router to get the expert probabilities for each token.
        expert_probabilities = self.router(x)

        # Get the top expert for each token
        selected_experts = torch.argmax(expert_probabilities, dim=-1).unsqueeze(-1)
        # You can do TOP-K routing if you are doing "No Token Left Behind" routing.

        outputs = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            selected_tokens = torch.nonzero(selected_experts == expert_idx).squeeze(-1)[:, 0]
            if len(selected_tokens) > 0:

                # The big change is here
                if selected_tokens.numel() > expert_capacity:
                    selected_tokens = randomly_select(selected_tokens, expert_capacity)
                # we randomly pass a expert_capacity numbers of tokens through this expert.
               
                # Same as before - forward pass the selected tokens through the expert.
                x_i = x[selected_tokens]
                expert_i = self.experts[expert_idx]
                out_i = expert_i(x_i)

                # Same as before - scale the output of the expert by the gating scores.
                gating_scores = expert_probabilities[selected_tokens, expert_idx]
                gating_scores = gating_scores.unsqueeze(1)
                out_i = out_i * gating_scores

                # Same as before - save the output. 
                outputs[selected_tokens] += out_i

        return outputs

SwitchTransformer = BasicSwitchTransformer(embed_size=1024, num_experts=8, capacity_factor=1)
x = torch.randn(10, 1024)
SwitchTransformer(x)
