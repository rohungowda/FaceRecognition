import torch
from KP_RPE import KR_RPE
from Embedding_Layer import LearnedEmbeddings
from transformer import Transformer
from ArcFace import ArcFace
from Constants import L

class FaceRec(torch.nn.Module):
    def __init__(self, position_embed, MeshGrid, DistanceMatrix, K, m):
        super().__init__()

        self.KR_RPE_layer = KR_RPE(MeshGrid, DistanceMatrix, K)
        self.embedding_layer = LearnedEmbeddings(position_embed)
        self.transformer_layer = Transformer()
        self.arcface_layer = ArcFace(m)

    def forward(self, patches, keypoints):
        b_matrix = self.KR_RPE_layer(keypoints)
        embeddings = self.embedding_layer(patches)

        for _ in range(L): # use 1 for cpu testing
            print(". transformer layer")
            embeddings = self.transformer_layer(embeddings, b_matrix)

        logits = self.arcface_layer(embeddings[:,0,:])

        return logits