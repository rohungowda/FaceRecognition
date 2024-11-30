import os
import torch
import pickle
from Constants import M, MODEL_SAVE_PATH
from helper import PrecomputeMeshGrid, ComputePatches, PrecomputeDistances, PrecomputePositionalEncoding
from Face_rec import FaceRec

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_loss_path = os.path.join(MODEL_SAVE_PATH, f"training_losses.pkl")
testing_loss_path = os.path.join(MODEL_SAVE_PATH, f"testing_losses.pkl")


def load_losses(type,file_path):
    try:
        with open(file_path, 'rb') as f:
            losses = pickle.load(f)
        return losses[type]
    except FileNotFoundError:
        return []
    
print(load_losses("Training_losses", training_loss_path))
print(load_losses("Testing_losses", testing_loss_path))


MeshGrid = PrecomputeMeshGrid()
DistanceMatrix, K = PrecomputeDistances()
position_embed = PrecomputePositionalEncoding()

model = FaceRec(position_embed, MeshGrid, DistanceMatrix, K, M)

e = 4

model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, f"FaceRecModel_{e}.pth"), weights_only=True))

model.eval()

with torch.no_grad():
    
    test_image = torch.randn((1,3,256,256), dtype=torch.float64)
    keypoints = torch.randn((1,5,2), dtype=torch.float64)

    patches = ComputePatches(test_image)

    logits = model(patches, keypoints)

    print(torch.argmax(logits))
