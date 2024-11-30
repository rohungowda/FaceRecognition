import torch
from Constants import PATCH_SIZE, IMAGE_SIZE, N, EMBEDDING_DIM


'''
Easily understand defenition
(257,1) -> (boradcasts the 1 element 1024 times for each row) -> (257 x 1024) / dividing works by dividing each row, -> 2, 2, 2, 2, 2, 2
(1,1024) -> (broadcasts the 1024 vector 257 times) -> (257 x 1024)                                      by each row  -> [ 0, 1, 2, 3, 4]             
'''

def PrecomputePositionalEncoding():
    # sinsuodial functions +1 without SeqPool Layer
    positions = torch.arange(int(N)).unsqueeze(1).double()
    embedding_positions = torch.arange(EMBEDDING_DIM).unsqueeze(0).double()
    even_indices= (embedding_positions[embedding_positions % 2 == 0]).int()
    odd_indices = (embedding_positions[embedding_positions % 2 != 0]).int()

    encoded_positions = positions / (10000 ** ((2.0 * embedding_positions) / EMBEDDING_DIM))

    encoded_positions[:, even_indices]= torch.sin(encoded_positions[:, even_indices])
    encoded_positions[:, odd_indices]= torch.cos(encoded_positions[:, odd_indices])

    return encoded_positions

def PrecomputeDistances():
    n = IMAGE_SIZE / PATCH_SIZE
    basis = torch.arange(PATCH_SIZE // 2, IMAGE_SIZE, PATCH_SIZE)
    pos = (torch.stack([basis.repeat_interleave(int(n)), torch.concat([basis] * int(n), dim=0)], dim=0)).T    
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    matrix = (torch.floor(torch.sqrt(torch.sum(diff**2, dim=2)))).long() # int64
    return matrix, torch.max(matrix).item() + 1

def ComputePatches(img):
    if int(IMAGE_SIZE) % PATCH_SIZE != 0:
        print("Patch size does not evenly divide by Image size")
        exit(1)
    unfold = torch.nn.Unfold((PATCH_SIZE,PATCH_SIZE), stride=PATCH_SIZE)
    output = unfold(img)
    output = output.permute(0,2,1)
    return output


def PrecomputeMeshGrid():
    if int(IMAGE_SIZE) % PATCH_SIZE != 0:
        print("Patch size does not evenly divide by Image size")
        exit(1)

    n = IMAGE_SIZE / PATCH_SIZE
    basis = torch.arange(PATCH_SIZE // 2, IMAGE_SIZE, PATCH_SIZE)
    mesh = torch.stack([basis.repeat_interleave(int(n)), torch.concat([basis] * int(n), dim=0)], dim=0) * (1/IMAGE_SIZE)
    return mesh.T
