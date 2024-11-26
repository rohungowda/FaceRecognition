import torch
from Constants import EMBEDDING_DIM, N, ATTENTION_HEADS

class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.WQ = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), dtype=torch.float64))
        self.WK = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), dtype=torch.float64))
        self.WV = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), dtype=torch.float64))
        self.WO = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), dtype=torch.float64))
        self.division = EMBEDDING_DIM // ATTENTION_HEADS

    def forward(self, embeddings, B_matrix):
        
        Qh = (embeddings @ self.WQ).reshape(-1, ATTENTION_HEADS, int(N + 1),  self.division)
        Kh = (embeddings @ self.WK).reshape(-1, ATTENTION_HEADS, int(N + 1),  self.division).permute(0,1,3,2)
        Vh = (embeddings @ self.WV).reshape(-1, ATTENTION_HEADS, int(N + 1),  self.division)

        attention_matrix = ((Qh @ Kh) + torch.nn.functional.pad(B_matrix, (1,0,1,0))) / torch.sqrt(torch.tensor(self.division).double())
        attention_matrix = torch.softmax(attention_matrix, dim=-1)

        H = (attention_matrix @ Vh).permute(0,2,1,3).reshape(-1, int(N+1), EMBEDDING_DIM)

        MH = H @ self.WO

        return MH

