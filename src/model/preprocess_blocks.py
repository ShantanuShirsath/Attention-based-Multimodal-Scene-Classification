import torch.nn as nn
import torch.nn.functional as F
from src.exception import CustomException
from src.logger import logging
import os
import sys


class image_CNN_block(nn.Module):  #assume imput size 3,224,224)
    def __init__(self):
        super(image_CNN_block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 32, kernel_size= 3,padding=1)  
        self.pool1 = nn.MaxPool2d(2,2) 
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size= 3,padding=1) 
        self.pool2 = nn.MaxPool2d(2,2) 
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size= 3,padding=1) 
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels= 512, kernel_size= 3,padding=1) 
        self.pool4 = nn.MaxPool2d(2,2)
        
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
         
        self.fc1 = nn.Linear(in_features= 512*14*14, out_features= 256)
        self.drop1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(in_features=256, out_features= 128)
        self.drop2 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(in_features=128,out_features=64)
        

    def forward(self,x):
        try:
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            
            x = F.relu(self.conv4(x))
            x = self.pool4(x)
            #print(x.shape)
            
            x = self.flatten(x)
            
            x = F.relu(self.fc1(x))
            x = self.drop1(x)
            
            x = F.relu(self.fc2(x))
            x = self.drop2(x)
            
            x = self.fc3(x)
        
            return x  
        
        except Exception as e:
            raise CustomException(e,sys)
        
    

class sound_block(nn.Module):
    def __init__(self):
        super(sound_block,self).__init__()
        
        self.fc1 = nn.Linear(in_features= 103, out_features= 128)
        #self.drop1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(in_features=128, out_features= 512)
        #self.drop2 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(in_features=512, out_features= 256)
        #self.drop3 = nn.Dropout(p=0.3)
        
        self.out = nn.Linear(in_features=256, out_features= 64)  

    def forward(self,x):
        try:
            x = F.relu(self.fc1(x))
            #x = self.drop1(x)
            
            x = F.relu(self.fc2(x))
            #x = self.drop2(x)
            
            x = F.relu(self.fc3(x))
            #x = self.drop3(x)
            
            x = self.out(x)
            return x  
          
        except Exception as e:
            raise CustomException(e,sys)


class FC_final_block(nn.Module):
    def __init__(self):
        super(FC_final_block,self).__init__()
        self.fc1 = nn.Linear(in_features= 128, out_features= 256)
        #self.drop1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(in_features=256, out_features= 128)
        #self.drop2 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(in_features=128, out_features= 64)
        #self.drop3 = nn.Dropout(p=0.3)
        
        self.out = nn.Linear(in_features=64, out_features= 2)
        
    def forward(self,x):
        try:
            x = F.relu(self.fc1(x))
            #x = self.drop1(x)
            
            x = F.relu(self.fc2(x))
            #x = self.drop2(x)
            
            x = F.relu(self.fc3(x))
            #x = self.drop3(x)
            
            x = self.out(x)
        
            return x
        
        except Exception as e:
            raise CustomException(e,sys)