import torch
from Constants import EMBEDDING_DIM, PATCH_SIZE, CHANNELS


class LearnedEmbeddings(torch.nn.Module):
    def __init__(self, position_embed):
        super().__init__()

        self.W = torch.nn.Parameter(torch.empty(((PATCH_SIZE * PATCH_SIZE * CHANNELS),EMBEDDING_DIM), dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.W)
        self.position_embed = position_embed
    
    def forward(self, patches):
        Embeddings = patches @ self.W

        final_embeddings = Embeddings + self.position_embed

        return final_embeddings


