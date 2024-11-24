from Dataset import Image_Features_Dataset
from torch.utils.data import DataLoader
from Constants import TRAIN_FEATURES_CSV_PATH, TEST_FEATURES_CSV_PATH, CELEB_TRAINING_PATH
from helper import PrecomputeMeshGrid, ComputePatches

train_dataset = Image_Features_Dataset(TRAIN_FEATURES_CSV_PATH, CELEB_TRAINING_PATH)
test_dataset = Image_Features_Dataset(TEST_FEATURES_CSV_PATH, CELEB_TRAINING_PATH)

dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
for iteration, batch in enumerate(dataloader):
    image, label, keypoints = batch
    print("Iteration at " + str(iteration))
    patches = ComputePatches(image)
    print(patches.size())
