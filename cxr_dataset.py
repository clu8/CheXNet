import os

import pandas as pd
from PIL import Image
import torch


class CXRDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, data_csv, transform=None, patient_filter=None):
        '''
        patient_filter: A function which receives the patient ID as input and returns whether an image should be included in the dataset. 
                        Useful for splitting train into train/val.
        '''
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

        if patient_filter is not None:
            img_to_patient = {img_file: patient_id for img_file, patient_id in zip(df['Image Index'], df['Patient ID'])}
            self.img_files = [img_file for img_file in self.img_files if patient_filter(img_to_patient[img_file])]
            
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        return img, self.img_to_labels[img_file]

class PneumoniaDataset(CXRDataset):
    def __init__(self, *args):
        super(PneumoniaDataset, self).__init__(*args)
        self.labels = [int('Pneumonia' in self.img_to_labels[img_file]) for img_file in self.img_files]
    
    def __getitem__(self, idx):
        img, _ = super(PneumoniaDataset, self).__getitem__(idx)
        return img, torch.FloatTensor([self.labels[idx]])
