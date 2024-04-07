import matplotlib.pyplot as plt
import torch
import os
import sys
from src.exception import CustomException
from src.logger import logging


class graphics():
    
    def __init__(self,epoch_loss,epoch_val_loss,epoch_accuracy,epoch_val_accuracy):
        self.epoch_loss = epoch_loss
        self.epoch_val_loss = epoch_val_loss
        self.epoch_accuracy = epoch_accuracy
        self.epoch_val_accuracy = epoch_val_accuracy
        

    def generate_loss_curve(self):
        try :
            # Generate x-values for epochs
            epochs = range(1, len(self.epoch_loss) + 1)

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, self.epoch_loss, 'b-', label='Training Loss')
            plt.plot(epochs, self.epoch_val_loss, 'r-', label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.xticks(ticks=range(0, len(epochs) + 1, 50))
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            raise CustomException(e,sys)
        
    def generate_accuracy_curve(self):
        try:
            epochs = range(1, len(self.epoch_accuracy) + 1)

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, self.epoch_accuracy, 'b-', label='Training Accuracy')
            plt.plot(epochs, self.epoch_val_accuracy, 'r-', label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy %')
            plt.xticks(ticks=range(0, len(epochs) + 1, 50))
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            raise CustomException(e,sys)
        
def save_model(net,model_name):
    torch.save(net, model_name + '.pth')

    # Or save only the model's state_dict
    torch.save(net.state_dict(), model_name + '.pth')