import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
from Constants import IMAGE_SIZE


class Image_Features_Dataset(Dataset):
    
    def __init__(self, image_features_path, img_dir):
        self.landmarks = pd.read_csv(image_features_path)
        self.img_dir = img_dir
    
    def __len__(self):
        return len(self.landmarks)
    
    def __getitem__(self,idx):
        
        curr_img_path = os.path.join(self.img_dir, self.landmarks.iloc[idx, 0])
        image = read_image(curr_img_path)
        image = image.to(torch.float32) / 255.0
        label = torch.tensor(self.landmarks.iloc[idx, 1])
        
        #res = list(self.landmarks.iloc[idx, 2:])
        #res = [[res[r - 1], res[r]] for r in range(1,len(res),2)]
        #keypoints = torch.tensor(res, dtype=torch.float32) / IMAGE_SIZE

        return image, label
