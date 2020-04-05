import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
import torchvision
import torchvision.models as models
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import numpy as np
import torch.nn.functional as F
import matplotlib.image as mpimg
import argparse
from main_func import *

argsparse = argparse.ArgumentParser(description='train.py')

argsparse.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
argsparse.add_argument('--save_dir', dest="save_dir", action="store", default="./mod.pth")
argsparse.add_argument('--arch', dest="arch", action="store", default="resnet18", type = str)
argsparse.add_argument('--learning_rate', type=float, dest="learning_rate", action="store", default=0.001)
argsparse.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=256)
argsparse.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
argsparse.add_argument('--gpu', dest="gpu", action="store", default="True", type = str)



parser = argsparse.parse_args()
data_dir = parser.data_dir
save_dir = parser.save_dir
arch = parser.arch
lr = parser.learning_rate
hidden_units = parser.hidden_units
epochs = parser.epochs
gpu = parser.gpu


        
if gpu == 'False':
    gpu = False
else:
    gpu = True


dataloaders, image_datasets = load_data(data_dir)

num_labels = len(image_datasets['train'].classes)    
model = ModelSpecial(num_labels, arch, hidden_units)    
# mapping of classes to indices which you get from  the image dataset
model.class_to_idx =image_datasets['train'].class_to_idx


if gpu and torch.cuda.is_available():
        print('Using GPU')
        device = torch.device("cuda:0")
        model.cuda()
else:
    print('Using CPU')
    device = torch.device("cpu") 
    
model.to(device)
if arch=='resnet18':
    optimizer = optim.Adam(model.basemodel.fc.parameters(),lr=lr)
else:
    optimizer = optim.Adam(model.basemodel.classifier[6].parameters(),lr=lr)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, dataloaders, criterion, optimizer, scheduler, device=device, num_epochs=epochs)


torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_to_idx':model.class_to_idx,
            'num_labels':model.num_labels,
            'hidden_number':model.hidden_number,
            'arch':model.arch
}, save_dir)