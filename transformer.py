import torch
from Constants import EMBEDDING_DIM, N, SCALE
from MultiHeadAttention import MultiHeadAttention

class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.normalization_layer = torch.nn.LayerNorm(EMBEDDING_DIM, dtype=torch.float64)
        self.normalization_layer_2 = torch.nn.LayerNorm(EMBEDDING_DIM, dtype=torch.float64)

        self.attention_layer = MultiHeadAttention()
        
        # only turn on for gpu
        self.expand = torch.nn.Linear((int(N+1) * EMBEDDING_DIM), SCALE * (int(N+1) * EMBEDDING_DIM), dtype=torch.float64)
        self.relu_layer = torch.nn.ReLU()
        self.contract = torch.nn.Linear(SCALE * (int(N+1) * EMBEDDING_DIM), (int(N+1) * EMBEDDING_DIM), dtype=torch.float64)

    def forward(self, embeddings, B_matrix):
        normalized_embeddings = self.normalization_layer(embeddings)
        attention_embeddings = self.attention_layer(normalized_embeddings, B_matrix)
        skip_embed = embeddings + attention_embeddings

        normalized_2 = self.normalization_layer_2(skip_embed)
        
        flattened_tensor = torch.flatten(normalized_2, start_dim=1)
        expanded_embeddings = self.expand(flattened_tensor)
        relu_embeddings = self.relu_layer(expanded_embeddings)
        contracted_embeddings = self.contract(relu_embeddings)
        reshaped_embeddings = torch.reshape(contracted_embeddings, (-1,int(N+1), EMBEDDING_DIM))


        final_embeddings = skip_embed + reshaped_embeddings

        return final_embeddings
    