CELEB_IDENTITY_PATH = "../Anno/identity_CelebA.txt"
CELEB_IMAGES_ALIGN_PATH = "../img_align_celeba"
CELEB_TRAINING_PATH = "../features/imgs"
TRAIN_FEATURES_CSV_PATH = "../features/train_features.csv"
TEST_FEATURES_CSV_PATH = "../features/test_features.csv"

IMAGE_SIZE = float(256.0)
RANDOM_SEED = 47
TEST_SIZE = 0.2
PATCH_SIZE = 128 # Change to 64 for actual model
N = (IMAGE_SIZE / PATCH_SIZE) *  (IMAGE_SIZE / PATCH_SIZE)

ATTENTION_HEADS = 16
ATTENTION_LAYERS = 1
EMBEDDING_DIM = 768
CHANNELS = 3

SCALE = 2 # this is fine
L = 12
CLASSES = 17
SUB_CENTERS = 3
M = 0.5