import numpy as np
import pandas as pd
import torch
import torch.utils
from torch.utils.data import Dataset
import torch.utils.data
from PIL import Image
import os
import sys
from src.exception import CustomException
from src.logger import logging


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img_path = self.data.iloc[idx, 1]
            #print("Image Path:", img_path)
            image = Image.open(r"B:\Projects\Scene_Classifier\src\data"+ img_path).convert('RGB')
            # Ensure that images are RGB
            #print(self.data.iloc[:, 1:104].dtypes)
            #print("Missing values in numeric columns:", self.data.iloc[:, 1:104].isnull().sum().sum())
            numeric_columns = self.data.iloc[idx, 2:-4].values.astype(np.float32)
            sound_data = torch.tensor(numeric_columns, dtype=torch.float32)
            label = torch.tensor(self.data.iloc[idx, -1])
            if self.transform:
                image = self.transform(image)

            return image, sound_data,label
        
        except Exception as e:
            raise CustomException(e,sys)
    
class Load_data():
    def __init__(self,dataset,batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset)
    
    def load(self):
        try:
            trainset,testset = torch.utils.data.random_split(self.dataset,[16000,1252])
            trainset, valset = torch.utils.data.random_split(trainset, [14000, 2000])
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,shuffle=True)
            testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,shuffle=True)
            valloader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size,shuffle=True)

            return trainloader,testloader,valloader
        
        except Exception as e:
            raise CustomException(e,sys)