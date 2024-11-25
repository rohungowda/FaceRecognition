import torch
from Dataset import Image_Features_Dataset
from torch.utils.data import DataLoader
from Constants import TRAIN_FEATURES_CSV_PATH, TEST_FEATURES_CSV_PATH, CELEB_TRAINING_PATH
from helper import PrecomputeMeshGrid, ComputePatches, PrecomputeDistances
from KP_RPE import KR_RPE

MeshGrid = PrecomputeMeshGrid()
DistanceMatrix, K = PrecomputeDistances()

train_dataset = Image_Features_Dataset(TRAIN_FEATURES_CSV_PATH, CELEB_TRAINING_PATH)
test_dataset = Image_Features_Dataset(TEST_FEATURES_CSV_PATH, CELEB_TRAINING_PATH)

dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

KR_RPE_layer = KR_RPE(MeshGrid, DistanceMatrix, K)



for iteration, batch in enumerate(dataloader):
    image, label, keypoints = batch

    print("Iteration at " + str(iteration))
    
    patches = ComputePatches(image)
    print(patches.size())

    break
    B_matrix = KR_RPE_layer(keypoints)
    print(B_matrix.size())
    # test loss REMOVE WHEN DONE
    loss = torch.tensor(5.0, requires_grad=True)
    loss.backward()
    break
