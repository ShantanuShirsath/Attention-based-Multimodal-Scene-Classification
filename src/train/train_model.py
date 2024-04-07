import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from src.data.make_dataset import CustomDataset,Load_data
from src.model import Multimodal_attention
from src.train import train_test
from src import utils
from src.logger import logging

# Create Dataset and Dataloaders
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust the size as needed
    transforms.ToTensor()  # Adjust mean and std if needed
])
batch_size = 64
dataset = CustomDataset("B:\Projects\Scene_Classifier\src\data\dataset_mod.csv", transform )
load_data = Load_data(dataset,batch_size=batch_size)
trainloader,testloader,valloader = load_data.load()


# Define Network and network parameters
net = Multimodal_attention()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss
optimizer = torch.optim.Adam(net.paramerters(),lr = 0.00001)
net.to(device)
epoch = 20

traintest = train_test(trainloader=trainloader,testloader=testloader,valloader=valloader,net=net,device=device,optimizer=optimizer,criterion=criterion,batch_size = batch_size)

epoch_loss = []*epoch 
epoch_Accuracy = []*epoch
epoch_val_loss = []*epoch 
epoch_val_Accuracy = []*epoch


for epoch_index in range(epoch):
    print(f'Epoch: {epoch_index}\n')

    train_loss, train_accuracy = traintest.train_one_epoch()
    val_loss, val_accuracy = traintest.validate_one_epoch()

    # Append the values to the lists
    epoch_loss.append(train_loss)
    epoch_Accuracy.append(train_accuracy)

    # Optionally, you can also append validation values
    epoch_val_loss.append(val_loss)
    epoch_val_Accuracy.append(val_accuracy)

print('Finished Training')

# graphics = utils.graphics(epoch_loss,epoch_val_loss,epoch_Accuracy,epoch_val_Accuracy)
# graphics.generate_accuracy_curve()
# graphics.generate_loss_curve()

logging.info(f"test accuracy = {traintest.test_one_epoch()}")
utils.save_model(net,"model_1")