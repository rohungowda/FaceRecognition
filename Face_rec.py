import torch
from KP_RPE import KR_RPE
from Embedding_Layer import LearnedEmbeddings
from transformer import Transformer
from ArcFace import ArcFace
from Constants import L
from SeqPool import SeqPool

class FaceRec(torch.nn.Module):
    def __init__(self, position_embed, MeshGrid, DistanceMatrix, K, m):
        super().__init__()

        self.KR_RPE_layer = KR_RPE(MeshGrid, DistanceMatrix, K)
        self.embedding_layer = LearnedEmbeddings(position_embed)
        self.transformer_layer = Transformer()
        self.arcface_layer = ArcFace(m)
        self.sequential_layer = SeqPool()

    def forward(self, patches, keypoints):
        b_matrix = self.KR_RPE_layer(keypoints)
        embeddings = self.embedding_layer(patches)

        for _ in range(L):
            embeddings = self.transformer_layer(embeddings, b_matrix)

        classification_embedding = self.sequential_layer(embeddings)
        classification_embedding = classification_embedding.squeeze(1)
        
        logits = self.arcface_layer(classification_embedding)

        return logits