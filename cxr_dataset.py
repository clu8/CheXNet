import os

import pandas as pd
from PIL import Image
import torch


class CXRDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, data_csv, transform=None):
        self.transform = transform
        
        self.img_dir = img_dir
        self.img_files = os.listdir(img_dir)
        
        self.img_to_labels = {}
        df = pd.read_csv(data_csv)
        for img_file, labels_str in zip(df['Image Index'], df['Finding Labels']):
            if labels_str == 'No Finding':
                labels = set()
            else:
                labels = set(labels_str.split('|'))
            self.img_to_labels[img_file] = labels
            
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        #print(img_file)
        img_path = os.path.join(self.img_dir, img_file)
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
            
        return img, self.img_to_labels[img_file]

class PneumoniaDataset(CXRDataset):
    def __getitem__(self, idx):
        img, labels = super(PneumoniaDataset, self).__getitem__(idx)
        return img, int('Pneumonia' in labels)
