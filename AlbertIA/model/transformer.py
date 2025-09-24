# transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlbertIA(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.output_layer(x)
        return logits
