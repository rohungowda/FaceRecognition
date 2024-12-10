import os
import torch
import pickle
from Constants import MODEL_SAVE_PATH
import matplotlib.pyplot as plt



def calculate_accuracy(logits, labels):
    softmax_logits = torch.softmax(logits, dim=-1)
    _, predicted_classes = softmax_logits.max(dim=-1)
    correct_predictions = (predicted_classes == labels).sum().item()
    accuracy = float(correct_predictions) / float(len(labels))
    return accuracy * 100.0

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_loss_path = os.path.join(MODEL_SAVE_PATH, f"training_loss.pkl")
testing_loss_path = os.path.join(MODEL_SAVE_PATH, f"testing_loss.pkl")


def load_losses(type,file_path):
    try:
        with open(file_path, 'rb') as f:
            losses = pickle.load(f)
        return losses[type]
    except FileNotFoundError:
        return []

training_losses = load_losses("Training_losses",training_loss_path)
testing_losses = load_losses("Testing_losses",testing_loss_path)
print(training_losses)
epochs = [i for i in range(len(training_losses))]

plt.figure(figsize=(10, 6))
plt.plot(epochs, training_losses, label='Training Losses', marker='o', linestyle='-', color='blue')
plt.plot(epochs, testing_losses, label='Testing Losses', marker='s', linestyle='--', color='green')

# Add informational panel
info_text = "Arc Margin = 0.50\nWeight Decay=0.0\nL=8\nCNN_Embeddings=216\nT_init=1000"
plt.text(0.7 * len(epochs), 0.8 * max(max(training_losses), max(testing_losses)), 
         info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# Labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.title('Training and Testing Losses (1st Attempt)')
plt.legend()
plt.grid(True)

# Save the plot as an image
plt.savefig('1st_Attempt.png', dpi=300)
