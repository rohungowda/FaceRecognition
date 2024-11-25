import torch
from Constants import EMBEDDING_DIM, N
from MultiHeadAttention import MultiHeadAttention

class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.normalization_layer = torch.nn.LayerNorm(EMBEDDING_DIM, dtype=torch.float64)
        self.attention_layer = MultiHeadAttention()

    def forward(self, embeddings, B_matrix):
        normalized_embeddings = self.normalization_layer(embeddings)
        attention_embeddings = self.attention_layer(normalized_embeddings, B_matrix)

        return attention_embeddings
        