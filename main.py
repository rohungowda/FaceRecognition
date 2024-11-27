import torch
from Dataset import Image_Features_Dataset
from torch.utils.data import DataLoader
from Constants import TRAIN_FEATURES_CSV_PATH, TEST_FEATURES_CSV_PATH, CELEB_TRAINING_PATH, M
from helper import PrecomputeMeshGrid, ComputePatches, PrecomputeDistances, PrecomputePositionalEncoding
from Face_rec import FaceRec


if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MeshGrid = PrecomputeMeshGrid()
DistanceMatrix, K = PrecomputeDistances()
position_embed = PrecomputePositionalEncoding()

train_dataset = Image_Features_Dataset(TRAIN_FEATURES_CSV_PATH, CELEB_TRAINING_PATH)
#test_dataset = Image_Features_Dataset(TEST_FEATURES_CSV_PATH, CELEB_TRAINING_PATH)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = FaceRec(position_embed, MeshGrid, DistanceMatrix, K, M)
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
loss = torch.nn.CrossEntropyLoss()
epochs = 5


train_metric_check = 10


for e in range(epochs):
    print(f"Epoch {e}")
    running_loss = 0.0
    total_loss = 0.0

    for iteration, batch in enumerate(train_dataloader):
        image, label, keypoints = batch

        optimizer.zero_grad()

        image = image.to(device)
        label = label.to(device)
        keypoints = keypoints.to(device)

        patches = ComputePatches(image)
        
        logits = model(patches, keypoints)
        logits = logits.squeeze(-1)

        loss_value = loss(logits,label)
        loss_value.backward()

        optimizer.step()

        running_loss += loss_value.item()

        total_loss += running_loss

        if iteration % train_metric_check == (train_metric_check - 5):
            print(f"Metric Calculation at Iteration {iteration}")
            print(f"Iteration Average Loss {(running_loss/(train_metric_check))}")
            running_loss = 0.0

    print("----------------------------------------------------------------")
    print(f"Total Average Loss {(total_loss/len(train_dataloader))}")
    break
