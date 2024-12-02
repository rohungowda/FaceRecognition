import torch
from KP_RPE import KR_RPE
from Embedding_Layer import LearnedEmbeddings
from transformer import Transformer
from ArcFace import ArcFace
from Constants import L
from SeqPool import SeqPool
from ConvEmbedding import ConvEmbeddings
from Constants import EMBEDDING_DIM, CLASSES

#  MeshGrid, DistanceMatrix, K,
# keypoints
class FaceRec(torch.nn.Module):
    def __init__(self, position_embed, m):
        super().__init__()

        #self.KR_RPE_layer = KR_RPE(MeshGrid, DistanceMatrix, K)
        self.embedding_layer = LearnedEmbeddings(position_embed)
        self.transformer_layer = Transformer()
        #self.arcface_layer = ArcFace(m)
        self.sequential_layer = SeqPool()
        self.convEmbedding_layer = ConvEmbeddings()
        self.mlp_classification = torch.nn.Linear(EMBEDDING_DIM, CLASSES)

    def forward(self, vision_patches, cnn_patches):
        #b_matrix = self.KR_RPE_layer(keypoints)
        embeddings = self.embedding_layer(vision_patches)
        conv_attention = self.convEmbedding_layer(cnn_patches)

        for _ in range(L):
            embeddings = self.transformer_layer(embeddings, conv_attention)

        classification_embedding = self.sequential_layer(embeddings)
        classification_embedding = classification_embedding.squeeze(1)
        logits = self.mlp_classification(classification_embedding)
        
        #logits = self.arcface_layer(classification_embedding)

        return logits