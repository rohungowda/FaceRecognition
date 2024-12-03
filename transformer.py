import torch
from Constants import EMBEDDING_DIM, SCALE
from MultiHeadAttention import MultiHeadAttention

class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.normalization_layer = torch.nn.LayerNorm(EMBEDDING_DIM, dtype=torch.float32)
        self.normalization_layer_2 = torch.nn.LayerNorm(EMBEDDING_DIM, dtype=torch.float32)

        self.attention_layer = MultiHeadAttention()
        
        # only turn on for gpu
        self.expand = torch.nn.Linear(EMBEDDING_DIM, (SCALE * EMBEDDING_DIM), dtype=torch.float32)
        self.relu_layer = torch.nn.ReLU()
        self.contract = torch.nn.Linear((SCALE * EMBEDDING_DIM), (EMBEDDING_DIM), dtype=torch.float32)

    def forward(self, embeddings, B_matrix):

        normalized_embeddings = self.normalization_layer(embeddings)
        attention_embeddings = self.attention_layer(normalized_embeddings, B_matrix)
        skip_embed = embeddings + attention_embeddings

        normalized_2 = self.normalization_layer_2(skip_embed)
        
        expanded_embeddings = self.expand(normalized_2)
        relu_embeddings = self.relu_layer(expanded_embeddings)
        contracted_embeddings = self.contract(relu_embeddings)

        final_embeddings = skip_embed + contracted_embeddings

        return final_embeddings
    