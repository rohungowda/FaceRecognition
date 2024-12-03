import torch
from Constants import CLASSES, EMBEDDING_DIM, S, M

class ArcFaceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty((CLASSES,EMBEDDING_DIM), dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, X, labels):

        x_normalized_scalars = torch.linalg.vector_norm(X, dim=1).unsqueeze(1)
        norm_x = X / x_normalized_scalars
        columns_norm = torch.linalg.vector_norm(self.W, dim=1).unsqueeze(1)
        norm_W = (self.W / columns_norm).permute(1,0)
        cos_thetas = norm_x @ norm_W


        loss_logits = None

        if labels is not None:
            one_hot = torch.nn.functional.one_hot(labels, num_classes=CLASSES).squeeze(1)
            loss_logits = S * torch.cos(torch.arccos(cos_thetas) + (one_hot * M))

        classification_logits = S *cos_thetas

        return classification_logits, loss_logits

