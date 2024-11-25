import torch
from KP_RPE import KR_RPE
from Embedding_Layer import LearnedEmbeddings
from transformer import Transformer

class FaceRec(torch.nn.Module):
    def __init__(self, position_embed, MeshGrid, DistanceMatrix, K):
        super().__init__()

        self.KR_RPE_layer = KR_RPE(MeshGrid, DistanceMatrix, K)
        self.embedding_layer = LearnedEmbeddings(position_embed)
        self.transformer_layer = Transformer()

    def forward(self, patches, keypoints):
        b_matrix = self.KR_RPE_layer(keypoints)
        embeddings = self.embedding_layer(patches)

        transformer_embeddings = self.transformer_layer(embeddings, b_matrix)

        return transformer_embeddings