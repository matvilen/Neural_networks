import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image 
import json

BWSquares_ds_path = os.path.abspath(os.path.join(os.getcwd(), '..', '_datasets\\BWSquares'))
BWSquares_coords = os.path.join(BWSquares_ds_path, 'coords.json')

class BWSquare_DS(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.img_list = []
        with open(BWSquares_coords, 'r') as f:
            self.target_dict = json.load(f)
            
        for root, dirs, files in os.walk(BWSquares_ds_path):
            for file in files:
                full_path = os.path.join(BWSquares_ds_path, file)
                if file in self.target_dict:
                    self.img_list.append((full_path, self.target_dict[file]))

        self.len_dataset = len(self.img_list)

        

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):


        if self.transform is not None:
            img = Image.open(self.img_list[index][0])
            img = self.transform(img)
            coord = torch.tensor(self.img_list[index][1], dtype=torch.float32)

        else:
            img = np.array(Image.open(self.img_list[index][0]))
            coord = np.array(self.img_list[index][1])
        
        return img, coord
        