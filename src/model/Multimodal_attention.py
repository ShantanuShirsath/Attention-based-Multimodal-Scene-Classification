from src.model import preprocess_blocks
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import sys
from src.exception import CustomException
from src.logger import logging

class Multi_Modal_attention(nn.Module):
    def __init__(self):
        super(Multi_Modal_attention,self).__init__()
        self.embed_dim = 64 
        self.heads = 8
        self.Sound_block = preprocess_blocks.sound_block()
        self.Image_CNN = preprocess_blocks.image_CNN_block()
        self.attention_image = nn.MultiheadAttention(embed_dim= self.embed_dim, num_heads=self.heads)
        self.attention_sound = nn.MultiheadAttention(embed_dim=self.embed_dim,num_heads= self.heads)
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.embed_dim,num_heads=self.heads)
        self.FC_final_block = preprocess_blocks.FC_final_block()
    
    def forward(self,image_input,sound_input):
        try:

            image_cnn_out = self.Image_CNN(image_input)
            sound_out = self.Sound_block(sound_input)
            #print("Before sound:",sound_out.shape)
            image_cnn_out = torch.unsqueeze(image_cnn_out,dim=-1)
            image_cnn_out = image_cnn_out.permute(0,2,1)
            sound_out = torch.unsqueeze(sound_out,dim=-1)
            sound_out = sound_out.permute(0,2,1)
            #print("Final sound:",sound_out.shape)
            
            image_self_attention,_ = self.attention_image(image_cnn_out,image_cnn_out,image_cnn_out)
            sound_self_attention,_ = self.attention_sound(sound_out,sound_out,sound_out)
            
            #print(image_self_attention.shape)
            #print(sound_self_attention.shape)
            
            image_cross_attention,_ = self.cross_attention(image_self_attention,sound_out,sound_out)
            sound_cross_attention,_ = self.cross_attention(sound_self_attention,image_cnn_out,image_cnn_out)
            
            image_cross_attention = image_cross_attention.view(image_cross_attention.size(0), -1)
            sound_cross_attention = sound_cross_attention.view(sound_cross_attention.size(0), -1)
            #print(image_cross_attention.shape)
            #print(sound_cross_attention.shape)
            concat = torch.cat((image_cross_attention, sound_cross_attention), dim=1)
            #print(concat.shape)
            x = self.FC_final_block(concat)
            return x
        
        except Exception as e:
            raise CustomException(e,sys)
