CELEB_IDENTITY_PATH = "../Anno/identity_CelebA.txt"
CELEB_IMAGES_ALIGN_PATH = "../img_align_celeba"
CELEB_TRAINING_PATH = "../new_features/imgs" # change between features and new_features
TRAIN_FEATURES_CSV_PATH = "../new_features/train_features.csv" # change between features and new_features
TEST_FEATURES_CSV_PATH = "../new_features/test_features.csv" # change between features and new_features
MODEL_SAVE_PATH = "../saved_models/"

IMAGE_SIZE = float(256.0)
RANDOM_SEED = 47
TEST_SIZE = 0.2
PATCH_SIZE = 32 
N = (IMAGE_SIZE / PATCH_SIZE) *  (IMAGE_SIZE / PATCH_SIZE)

ATTENTION_HEADS = 16
EMBEDDING_DIM = 768
CHANNELS = 3

SCALE = 2 # this is fine
L = 1 # CHANGE THIS TO 12
CLASSES = 450 # 0 - 449


SUB_CENTERS = 3
M = 0.5

BATCH_SIZE = 64
CNN_EMBEDDING = 256
CHUNK_SIZE = 16
S = 64.0
# ATTENTION_LAYERS = 1