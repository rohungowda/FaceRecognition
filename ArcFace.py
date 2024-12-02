import torch
from Constants import CLASSES, SUB_CENTERS, EMBEDDING_DIM, S

class ArcFace(torch.nn.Module):
    def __init__(self, m):
        super().__init__()

        self.m = m
        self.W = torch.nn.Parameter(torch.randn((CLASSES, SUB_CENTERS, EMBEDDING_DIM), dtype=torch.float32))
        self.pool_layer = torch.nn.MaxPool1d(SUB_CENTERS)

    def forward(self, X):
        #print(X.min(), X.max(), X.mean())

        norm_X = X.norm(p=2, dim=1)
        norm_W = self.W.norm(p=2)

        subCenter_angles = ((self.W @ X.T).permute(2,0,1)) / (norm_X * norm_W).view(-1,1,1)

        max_res = self.pool_layer(subCenter_angles)
        max_res = torch.clamp(max_res, min=-1.0, max=1.0)

        angular_margins = torch.cos(torch.arccos(max_res) + self.m) * S

        logits = torch.softmax(angular_margins, dim=1) # cross Entropy uses loss maybe need to remove?

        #print(logits[0])

        return logits
    

X = torch.randn((1, EMBEDDING_DIM))
labels = torch.randint(5, (5,1))
W = torch.nn.Parameter(torch.randn((5,EMBEDDING_DIM), dtype=torch.float32))
one_hot = torch.nn.functional.one_hot(labels, num_classes=5)


print(X.size())
print(labels)
print(W.size())
print(one_hot)