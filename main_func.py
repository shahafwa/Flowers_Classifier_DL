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
import json

class ModelSpecial(nn.Module):


    def __init__(self, num_labels = 102,class_to_idx= None, hidden_number=256, arch = 'resnet18'):
        super(ModelSpecial, self).__init__()
        if arch == 'resnet18':
            self.basemodel = models.resnet18(pretrained=True)
        elif arch == 'vgg19' or arch == 'alexnet':
            self.basemodel = models.vgg19(pretrained=True)
        else:
            raise ValueError('didnt excpect {} network architechture'.format(arch))

        # Freeze its parameters
        for param in self.basemodel.parameters():
            param.requires_grad = False
        
        if arch == 'resnet18':
            self.basemodel.fc = nn.Sequential(
            nn.Linear(512, hidden_number),
            nn.BatchNorm1d(hidden_number),
            nn.ReLU(True),
            nn.Linear(hidden_number, int(hidden_number/2)),
            nn.BatchNorm1d(int(hidden_number/2)),
            nn.ReLU(True),
            nn.Linear(int(hidden_number/2), num_labels))
            
        elif arch == 'vgg19' or arch == 'alexnet':
            self.basemodel.classifier[6] = nn.Sequential(
            nn.Linear(4096 , hidden_number),
            nn.BatchNorm1d(hidden_number),
            nn.ReLU(True),
            nn.Linear(hidden_units, int(hidden_number/2)),
            nn.BatchNorm1d(int(hidden_number/2)),
            nn.ReLU(True),
            nn.Linear(int(hidden_number/2), num_labels))
                    
        
        
        self.class_to_idx = class_to_idx
        self.num_labels = num_labels
        self.hidden_number = hidden_number
        self.arch = arch

            
    def forward(self, x):
        x = self.basemodel(x)
        return x

def load_data(data_dir):
    if isinstance(data_dir, list):
        data_dir = data_dir[0]
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(60),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ]),
    'val': transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224), transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])}


    # TODO: Load the datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                     'val': datasets.ImageFolder(valid_dir, transform=data_transforms['val']),
                     'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders =  {'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                     'val': DataLoader(image_datasets['val'], batch_size=64, shuffle=True),
                     'test': DataLoader(image_datasets['test'], batch_size=64)}
                     
    return dataloaders, image_datasets
    
    
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            count = 0
            

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    count += len(inputs)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            



            epoch_loss = running_loss / (len(dataloaders[phase])*64)
            epoch_acc = running_corrects.double() / (len(dataloaders[phase])*64)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    
    transformer = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(), #need to transform to tensor before normalize
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])])
    im = transformer(im)
    
    return im



def predict(image_path, model, topk=5, cat_to_name = None, names=False, device='cuda:0'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    im =process_image(image_path).unsqueeze(0).to(device)
    model= model.eval().to(device)
    output = model.forward(im)
    
    prob, labels = torch.topk(output, topk)
    prob = torch.nn.functional.softmax(prob, dim=1)[0].cpu()
    labels = labels.data.cpu().numpy()[0].tolist()
    
    
    
    class_idx = {val: i for i, val in model.class_to_idx.items()}
    
    if names:
        classes = [cat_to_name[class_idx[x]] for x in labels]
    else:
        classes = [class_idx[x] for x in labels]
        
            
    return prob,classes

def load_model(path):
    checkpoint = torch.load(path)
    class_to_idx = checkpoint['class_to_idx']
    num_labels = checkpoint['num_labels']
    hidden_number = checkpoint['hidden_number']
    arch = checkpoint['arch']
    model = ModelSpecial(num_labels = num_labels,class_to_idx= class_to_idx, hidden_number=hidden_number, arch = arch)
    model.load_state_dict(checkpoint['model_state_dict'])
    if arch=='resnet18':
        optimizer = optim.Adam(model.basemodel.fc.parameters())
    else:
        optimizer = optim.Adam(model.basemodel.classifier[6].parameters())
        
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model,optimizer
