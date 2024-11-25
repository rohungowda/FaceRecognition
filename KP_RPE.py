import torch
from Constants import N, ATTENTION_HEADS

class KR_RPE(torch.nn.Module):
    def __init__(self, MeshGrid, DistanceMatrix, K):
        super().__init__()

        # meshGrid is an (nxn) x 2 in this case (256 x 2) which gets converted to shape (64 x 5 x 2)
        self.mesh_grid = MeshGrid.unsqueeze(1).expand(-1, 5, -1)
        # Precomputed Distance matrix of 256 x 256
        self.distance_matrix = DistanceMatrix
        # Max number of distance buckets
        self.K = int(K)
        # Weights matrix of 10 x 3 * k
        self.W = torch.nn.Parameter(torch.randn((10,(ATTENTION_HEADS * K)), dtype=torch.float64))
        # indices to use, of shape 3 x 256 x 256
        self.indices = self.distance_matrix.unsqueeze(0).expand(ATTENTION_HEADS,-1,-1)
    
    def forward(self, keypoints):

        batch_size = keypoints.size(0)

        # keypoints is a 5 by 2 matrix which gets covnerted to 4 x 256 x 5 x 2
        converted_keypoints = keypoints.unsqueeze(1).expand(-1, int(N), -1, -1)
        mesh_grid = self.mesh_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # D subtracts the mesh_gird - keypoints and reshape to batch_size x 256 x 10
        D = torch.reshape((mesh_grid - converted_keypoints), (-1,int(N), 10))
        # (256 x 10) @ (256 x 3*K)
        A = D @ self.W
        # reshape (256 x 3*K) to (3, 256, K)
        A = torch.reshape(A,(-1, ATTENTION_HEADS, int(N), self.K))
        # uses the indices of the 3 x 256 x 256 matrix and applies it to the 3 x 256 x k matrix where the 0th dimesnion applies to head, the 1th for row and the indices to the 2nd
        indices = self.indices.unsqueeze(0).expand(batch_size, -1, -1, -1)
        B = torch.gather(A, dim=3, index=indices)
        # returns a (batch_size x 3 x 256 x 256 matrix)
        return B



# ** basically make sure you are getting diferent values or doing the operation batch wise and not applying th same batch over and over