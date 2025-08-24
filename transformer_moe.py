# This code covers the blueprint of a basic DenseMixtureOfExperts model.
import torch.nn as nn
import torch.nn.functional as F
from denseMoe import DenseMixtureOfExperts #or one of the sparse-moes

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadLatentAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

    def forward(self, x, mask):
        ## Attention code
        ## Check out https://gist.github.com/avbiswas/653b29d96cfd21b6f4739d61d9a2f573
        return x

class BasicGPT(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers):
        super(BasicGPT, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)

        # Initialize all decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(embed_size)
                for _ in range(num_layers)
            ]
        )
    
    def forward(self, x):

        # Input: [B, S], dtype: int64
        x = self.embedding_layer(x)
        # Output: [B, S, D], dtype: float32

        # Loop through decoder layers
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x)
        # Output: [B, S, D], dtype: float32
        
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size):
        super(TransformerDecoderLayer, self).__init__()
        self.attentionLayer = MultiHeadLatentAttention(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.dense_moe = DenseMixtureOfExperts(embed_size, num_experts=8) #or one of the sparse-moes
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # Runs one decoder layer

        batch_size, seq_length, embed_size = x.shape

        # Run attention layer for contextualizing tokens
        attention_output = self.attentionLayer(x)

        # Residual connection 1
        x = self.norm1(x + attention_output)
        
        # Flatten the tensor from [B, S, D] to [B * S, D]
        x_flat = x.view(batch_size * seq_length, embed_size)

        # Run dense MOE layer to replace the feedforward network
        ff_output = self.dense_moe(x_flat)

        # Reshape the tensor from [B * S, D] to [B, S, D]
        ff_output = ff_output.view(batch_size, seq_length, embed_size)

        # Residual connection 2
        x = self.norm2(x + ff_output)

        return x