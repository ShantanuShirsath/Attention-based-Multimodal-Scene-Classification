import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging


class train_test(nn.Module):
    def __init__(self,trainloader,valloader,testloader,device,optimizer,net,criterion,batch_size):
        super(train_test,self).__init__()
        self.trainloader = trainloader
        self.device = device
        self.optimizer = optimizer
        self.net = net
        self.criterion = criterion
        self.batch_size = batch_size
        self.valloader = valloader
        self.testloader = testloader
        
    
    def train_one_epoch(self):
        try:

            self.net.train(True)

            running_loss = 0.0
            running_accuracy = 0.0
            Epoch_loss = []
            Epoch_accuracy = []
            
            for batch_index, data in enumerate(self.trainloader):
                image,sound, labels = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.net(image,sound) # shape: [batch_size, 10]
                correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
                running_accuracy += correct / self.batch_size

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                if batch_index % 50 == 49:
                    avg_loss_across_batches = running_loss / 50
                    avg_acc_across_batches = (running_accuracy / 50) * 100
                    print('Batch {0}, Loss: {1:.5f}, Accuracy: {2:.3f}%'.format(batch_index+1,
                                                                        avg_loss_across_batches,
                                                                        avg_acc_across_batches))
                    running_loss = 0.0
                    running_accuracy = 0.0
                    Epoch_loss.append(avg_loss_across_batches)
                    Epoch_accuracy.append(avg_acc_across_batches)

            print()
            
            return  sum(Epoch_loss)/len(Epoch_loss) , sum(Epoch_accuracy)/len(Epoch_accuracy)
        
        except Exception as e:
            raise CustomException(e,sys)

    
    def validate_one_epoch(self):

        try:

            self.net.train(False)
            running_loss = 0.0
            running_accuracy = 0.0
            Epoch_loss = []
            Epoch_accuracy = []

            for i, data in enumerate(self.valloader):
                image,sound, labels = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)

                with torch.no_grad():
                    outputs = self.net(image,sound) # shape: [batch_size, 10]
                    correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
                    running_accuracy += correct / self.batch_size
                    loss = self.criterion(outputs, labels) # One number, the average batch loss
                    running_loss += loss.item()

            avg_loss_across_batches = running_loss / len(self.valloader)
            avg_acc_across_batches = (running_accuracy / len(self.valloader)) * 100

            print('Val Loss: {0:.5f}, Val Accuracy: {1:.3f}%'.format(avg_loss_across_batches,
                                                                    avg_acc_across_batches))
            print('***************************************************')
            print()
            Epoch_loss.append(avg_loss_across_batches)
            Epoch_accuracy.append(avg_acc_across_batches)
            
            return  Epoch_loss,Epoch_accuracy
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def test_one_epoch(self):
        try:

            self.net.train(False)
            running_loss = 0.0
            running_accuracy = 0.0

            for i, data in enumerate(self.testloader):
                image,sound, labels = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)

                with torch.no_grad():
                    outputs = self.net(image,sound) # shape: [batch_size, 10]
                    correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
                    running_accuracy += correct / self.batch_size
                    loss = self.criterion(outputs, labels) # One number, the average batch loss
                    running_loss += loss.item()

            avg_loss_across_batches = running_loss / len(self.testloader)
            avg_acc_across_batches = (running_accuracy / len(self.testloader)) * 100

            print('test Loss: {0:.5f}, test Accuracy: {1:.3f}%'.format(avg_loss_across_batches,
                                                                    avg_acc_across_batches))
            print('***************************************************')
            print()

            return avg_loss_across_batches,avg_acc_across_batches
        
        except Exception as e:
            raise CustomException(e,sys)