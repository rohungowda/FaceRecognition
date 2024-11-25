import torch
from Constants import N, EMBEDDING_DIM, PATCH_SIZE, CHANNELS

class LearnedEmbeddings(torch.nn.Module):
    def __init__(self, position_embed):
        super().__init__()

        self.W =  torch.nn.Parameter(torch.randn(((PATCH_SIZE * PATCH_SIZE * CHANNELS),EMBEDDING_DIM), dtype=torch.float64))
        self.position_embed = position_embed


Patches = torch.randn((4,int(N),768), dtype=torch.float64)

W = torch.nn.Parameter(torch.randn(((PATCH_SIZE * PATCH_SIZE * CHANNELS),EMBEDDING_DIM), dtype=torch.float64))
Embeddings = Patches @ W
classification_token = torch.randn((Embeddings.size(0),1,EMBEDDING_DIM), dtype=torch.float64)
Embedding_w_classification = torch.cat((classification_token, Embeddings), dim=1)

# positional encodings now

print(Embedding_w_classification.size())