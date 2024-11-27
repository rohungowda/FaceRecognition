import torch
from Constants import CLASSES, SUB_CENTERS, EMBEDDING_DIM

class ArcFace(torch.nn.Module):
    def __init__(self, m):
        super().__init__()

        self.m = m
        self.W = torch.nn.Parameter(torch.randn((CLASSES, SUB_CENTERS, EMBEDDING_DIM), dtype=torch.float64))
        self.pool_layer = torch.nn.MaxPool1d(SUB_CENTERS)

    def forward(self, X):
        norm_X = X.norm(p=2, dim=1)
        norm_W = self.W.norm(p=2)

        subCenter_angles = ((self.W @ X.T).permute(2,0,1)) / (norm_X * norm_W).view(-1,1,1)

        max_res = self.pool_layer(subCenter_angles)

        angular_margins = torch.cos(torch.arccos(max_res) + self.m)
        logits = torch.softmax(angular_margins, dim=1)

        return logits