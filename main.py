import torch
from Dataset import Image_Features_Dataset
from torch.utils.data import DataLoader
from Constants import TRAIN_FEATURES_CSV_PATH, TEST_FEATURES_CSV_PATH, CELEB_TRAINING_PATH
from helper import PrecomputeMeshGrid, ComputePatches, PrecomputeDistances, PrecomputePositionalEncoding
from Face_rec import FaceRec

MeshGrid = PrecomputeMeshGrid()
DistanceMatrix, K = PrecomputeDistances()
position_embed = PrecomputePositionalEncoding()

train_dataset = Image_Features_Dataset(TRAIN_FEATURES_CSV_PATH, CELEB_TRAINING_PATH)
test_dataset = Image_Features_Dataset(TEST_FEATURES_CSV_PATH, CELEB_TRAINING_PATH)

dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = FaceRec(position_embed, MeshGrid, DistanceMatrix, K)



for iteration, batch in enumerate(dataloader):
    image, label, keypoints = batch
    patches = ComputePatches(image)
    print("Iteration at " + str(iteration))
    
   # get b_matrix and embeddings - stage 1
    transformer_embedding = model(patches, keypoints)

    print(transformer_embedding.size())

    break
    # test loss REMOVE WHEN DONE
    loss = torch.tensor(5.0, requires_grad=True)
    loss.backward()
    break
