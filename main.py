import os
import torch
import pickle
from Dataset import Image_Features_Dataset
from torch.utils.data import DataLoader
from Constants import TRAIN_FEATURES_CSV_PATH, TEST_FEATURES_CSV_PATH, CELEB_TRAINING_PATH, MODEL_SAVE_PATH, BATCH_SIZE, INITIAL_T, T_MULT
from helper import ComputePatches, PrecomputePositionalEncoding
from Face_rec import FaceRec


def calculate_accuracy(logits, labels):
    softmax_logits = torch.softmax(logits, dim=-1)
    _, predicted_classes = softmax_logits.max(dim=-1)
    correct_predictions = (predicted_classes == labels).sum().item()
    accuracy = float(correct_predictions) / float(len(labels))
    return accuracy * 100.0

def save_losses(type, losses, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump({f'{type}': losses}, f)

gpu_id = '0'
if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(gpu_id)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

# Important Variables ******************************************
position_embed = PrecomputePositionalEncoding().to(device)

train_dataset = Image_Features_Dataset(TRAIN_FEATURES_CSV_PATH, CELEB_TRAINING_PATH)
test_dataset = Image_Features_Dataset(TEST_FEATURES_CSV_PATH, CELEB_TRAINING_PATH)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = FaceRec(position_embed)
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=INITIAL_T, T_mult=T_MULT)
loss = torch.nn.CrossEntropyLoss()

epochs = 100
metric_check = 10

training_losses = []
testing_losses = []

training_loss_path = os.path.join(MODEL_SAVE_PATH, f"training_loss.pkl")
testing_loss_path = os.path.join(MODEL_SAVE_PATH, f"testing_loss.pkl")

print(f"Number of Training Batches: {str(len(train_dataloader))}")
print(f"Number of Testing Batches: {str(len(test_dataloader))}") # 

for e in range(epochs):
    print(f"Epoch {e}")
    running_loss = 0.0
    total_loss = 0.0
    total_accuracy = 0.0
    c = 0

    print("---------------------------Training--------------------------------")
    for iteration, batch in enumerate(train_dataloader):

        image, labels = batch
        optimizer.zero_grad()

        vision_patches, conv_patches = ComputePatches(image)
        vision_patches = vision_patches.to(device)
        conv_patches = conv_patches.to(device)
        labels = labels.to(device)

        # *********************************** #

        classification_logits, loss_logits = model(vision_patches, conv_patches, labels)

        loss_value = loss(loss_logits,labels)
        loss_value.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss_value.item()

        total_loss += loss_value.item()

        if torch.isnan(loss_value):
            print("Loss is nan")
            exit(0)


        if iteration % metric_check == (metric_check - 1):
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Metric Calculation at Iteration {iteration}")
            print(f"Iteration Training Average Loss {(running_loss/(metric_check))}")
            accuracy = calculate_accuracy(classification_logits, labels)
            total_accuracy += accuracy
            print(f"Iteration Training Accuracy {accuracy}")

            running_loss = 0.0
            c += 1
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    print("----------------------------End Training--------------------------------")
    total_average_loss = (total_loss/len(train_dataloader))
    total_average_accuracy = (total_accuracy/c)

    print(f"Total Training Average Loss {total_average_loss}")
    print(f"Total Training Average Accuracy {total_average_accuracy}")

    training_losses.append(total_average_loss)

    save_losses("Training_losses", training_losses, training_loss_path)

    # saving model
    save_model = os.path.join(MODEL_SAVE_PATH, f"FaceRecModel_{e}.pth")
    torch.save(model.state_dict(), save_model)

    # testing iteration *************************************** testing iteration
    running_loss = 0.0
    total_loss = 0.0
    total_accuracy = 0.0
    c = 0

    print("-----------------------------Testing------------------------------")
    with torch.no_grad():
        for iteration, batch in enumerate(test_dataloader):
                image, labels = batch

                vision_patches, conv_patches = ComputePatches(image)
                vision_patches = vision_patches.to(device)
                conv_patches = conv_patches.to(device)
                labels = labels.to(device)

                # ************************************ #

                classification_logits, loss_logits = model(vision_patches, conv_patches,labels)

                loss_value = loss(classification_logits,labels)

                running_loss += loss_value.item()

                total_loss += loss_value.item()


                if iteration % metric_check == (metric_check - 1):
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print(f"Metric Calculation at Iteration {iteration}")
                    print(f"Iteration Testing Average Loss {(running_loss/(metric_check))}")
                    accuracy = calculate_accuracy(classification_logits, labels)
                    total_accuracy += accuracy
                    print(f"Iteration Testing Accuracy {accuracy}")

                    running_loss = 0.0
                    c += 1
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    print("--------------------------End Testing----------------------------------")
    total_average_loss = (total_loss/len(test_dataloader))
    total_average_accuracy = (total_accuracy/c)

    print(f"Total Testing Average Loss {total_average_loss}")
    print(f"Total Testing Average Accuracy {total_average_accuracy}")

    testing_losses.append(total_average_loss)

    save_losses("Testing_losses", testing_losses, testing_loss_path)

    print(f"End of Epoch {e}")

