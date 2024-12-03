import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from Constants import CELEB_IDENTITY_PATH, IMAGE_SIZE, TRAIN_FEATURES_CSV_PATH, CELEB_TRAINING_PATH, TEST_FEATURES_CSV_PATH, RANDOM_SEED, TEST_SIZE, CELEB_IMAGES_ALIGN_PATH
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from torchvision import transforms



def Process_Image(image_name):

    image_path = os.path.join(CELEB_IMAGES_ALIGN_PATH, image_name)
    image = Image.open(image_path)

    Resize_Function = transforms.Resize((int(IMAGE_SIZE), int(IMAGE_SIZE)))
    image_resized = Resize_Function(image)

    image_tensor = transforms.ToTensor()(image_resized) 
    image_np = image_tensor.permute(1, 2, 0).numpy()

    image_np = (image_np * 255).astype(np.uint8)
    image_save = Image.fromarray(image_np)
    image_save.save(os.path.join(CELEB_TRAINING_PATH, image_name))



def Process_Data():
    identity_lines = Path(CELEB_IDENTITY_PATH).read_text().splitlines()

    # Class_Number: list of images
    identity_hash = defaultdict(list)

    for line in identity_lines:
        value,key = line.split(' ')
        identity_hash[key].append(value)

    res = list(identity_hash.items())

    Limit = 27  # use 25 very usefull
    K_identity = list(filter(lambda x : len(x[1]) > Limit, res))
    

    columns = ["ImageFileName","Class"]

    df = pd.DataFrame(columns=columns).to_csv(TRAIN_FEATURES_CSV_PATH, index=False)

    for i, (Class, images) in enumerate(K_identity):
        print(str((i/len(K_identity)) * 100))
        for j,image in enumerate(images):
            Process_Image(image)
            data_point = [image, i]
            data_row = pd.DataFrame([data_point], columns=columns)
            data_row.to_csv(TRAIN_FEATURES_CSV_PATH, mode='a', index=False, header=False)

        print(f"Class: {i} completed")


def split_data():
    df = pd.read_csv(TRAIN_FEATURES_CSV_PATH)

    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df['Class'], random_state=RANDOM_SEED)

    train_df.to_csv(TRAIN_FEATURES_CSV_PATH, index=False)
    test_df.to_csv(TEST_FEATURES_CSV_PATH, index=False)

    print("Number of Total Samples: " + str(len(df)))
    print("Number of Training Samples: " + str(len(train_df)))
    print("Number of Testing Samples: " + str(len(test_df)))


print("0 for Processing data, 1 for splitting data, -1 for exit")
choice = input()

if choice == '0':
    Process_Data()
elif choice == '1':
    split_data()
else:
    exit(0)