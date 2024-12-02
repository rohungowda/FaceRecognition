import torch
from Constants import EMBEDDING_DIM, N


class SeqPool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_weights = torch.nn.Linear(EMBEDDING_DIM, 1, dtype=torch.float32)

    def forward(self, transformer_embeddings):
        output = torch.softmax(self.patch_weights(transformer_embeddings).permute(0,2,1), dim=2)
        classification_embedding = output @ transformer_embeddings
        
        return classification_embedding
        

