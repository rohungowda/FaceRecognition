import os
import torch
import pickle
from Dataset import Image_Features_Dataset
from torch.utils.data import DataLoader
from Constants import TRAIN_FEATURES_CSV_PATH, TEST_FEATURES_CSV_PATH, CELEB_TRAINING_PATH, M, MODEL_SAVE_PATH, BATCH_SIZE
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
test_dataset = Image_Features_Dataset(TEST_FEATURES_CSV_PATH, CELEB_TRAINING_PATH)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = FaceRec(position_embed, MeshGrid, DistanceMatrix, K, M)
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
loss = torch.nn.CrossEntropyLoss()
epochs = 5

training_losses = []
testing_losses = []

metric_check = 10


# ** test

training_loss_path = os.path.join(MODEL_SAVE_PATH, f"training_losses.pkl")
testing_loss_path = os.path.join(MODEL_SAVE_PATH, f"testing_losses.pkl")

# Function to save losses
def save_losses(type, losses, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump({f'{type}_losses': losses}, f)

print(len(train_dataloader)) # 113 
print(len(test_dataloader)) # 29

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

        patches = ComputePatches(image).to(device)
        
        logits = model(patches, keypoints)
        logits = logits.squeeze(-1)

        loss_value = loss(logits,label)
        loss_value.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss_value.item()

        total_loss += loss_value.item()

        # test * remove when done
        break


        if iteration % metric_check == (metric_check - 1):
            print(f"Loss Calculation at Iteration {iteration}")
            print(f"Iteration Training Average Loss {(running_loss/(metric_check))}")
            running_loss = 0.0

    print("----------------------------------------------------------------")
    total_average_loss = (total_loss/len(train_dataloader))
    print(f"Total Training Average Loss {total_average_loss}")

    training_losses.append(total_average_loss)
    save_losses("Training", training_losses, training_loss_path)

    # saving model
    save_model = os.path.join(MODEL_SAVE_PATH, f"FaceRecModel_{e}.pth")
    torch.save(model.state_dict(), save_model)

    # testing iteration *************************************** testing iteration
    running_loss = 0.0
    total_loss = 0.0

    with torch.no_grad():
        for iteration, batch in enumerate(test_dataloader):
                image, label, keypoints = batch

                image = image.to(device)
                label = label.to(device)
                keypoints = keypoints.to(device)

                patches = ComputePatches(image)
                
                logits = model(patches, keypoints)
                logits = logits.squeeze(-1)

                loss_value = loss(logits,label)

                running_loss += loss_value.item()

                total_loss += loss_value.item()


                # test * remove when done
                break


                if iteration % metric_check == (metric_check - 1):
                    print(f"Loss Calculation at Iteration {iteration}")
                    print(f"Iteration Testing Average Loss {(running_loss/(metric_check))}")
                    running_loss = 0.0

    print("----------------------------------------------------------------")
    total_average_loss = (total_loss/len(test_dataloader))
    print(f"Total Testing Average Loss {total_average_loss}")

    testing_losses.append(total_average_loss)
    save_losses("Testing", testing_losses, testing_loss_path)

    print(f"End of Epoch {e}")
    # ** testing
