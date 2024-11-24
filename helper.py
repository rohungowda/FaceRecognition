import torch
from Constants import PATCH_SIZE, IMAGE_SIZE

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
    mesh = torch.stack([basis.repeat_interleave(int(n)), torch.concat([basis] * int(n), dim=0)], dim=0) * (1/float(n*n))
    return mesh.T


patch_size = 2
image_size = 10
channels = 3
test = torch.zeros((channels, image_size, image_size))

count = 1
for i in range(0, image_size, patch_size):
    for j in range(0,image_size, patch_size):
        test[:,i:(i+patch_size), j:(j+patch_size)] = torch.full((channels, patch_size,patch_size),count)
        count += 1

test = torch.stack([test,test,test, test], dim=0)
unfold = torch.nn.Unfold((patch_size,patch_size), stride=patch_size)
output = unfold(test)
output = output.permute(0,2,1)



