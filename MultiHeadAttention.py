import torch
from Constants import EMBEDDING_DIM, N, ATTENTION_HEADS

class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.WQ = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), dtype=torch.float32))
        self.WK = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), dtype=torch.float32))
        self.WV = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), dtype=torch.float32))
        self.WO = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), dtype=torch.float32))
        self.division = EMBEDDING_DIM // ATTENTION_HEADS

    def forward(self, embeddings, B_matrix):
        
        Qh = (embeddings @ self.WQ).reshape(-1, ATTENTION_HEADS, int(N),  self.division)
        Kh = (embeddings @ self.WK).reshape(-1, ATTENTION_HEADS, int(N),  self.division).permute(0,1,3,2)
        Vh = (embeddings @ self.WV).reshape(-1, ATTENTION_HEADS, int(N),  self.division)

        attention_matrix = ((Qh @ Kh)) / torch.sqrt(torch.tensor(self.division))
        attention_matrix = attention_matrix + B_matrix
        attention_matrix = torch.softmax(attention_matrix, dim=-1)

        H = (attention_matrix @ Vh).permute(0,2,1,3).reshape(-1, int(N), EMBEDDING_DIM)

        MH = H @ self.WO

        return MH

