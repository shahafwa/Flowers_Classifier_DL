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
import json

argsparse = argparse.ArgumentParser(description='train.py')

argsparse.add_argument('im_path', nargs='*', action="store", default="./flowers/test/1/image_06752.jpg")
argsparse.add_argument('checkpoint', nargs='*', action="store", default="mod.pth")
argsparse.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
argsparse.add_argument('--category_names', dest="category_names", action="store", default='./cat_to_name.json')
argsparse.add_argument('--gpu', dest="gpu", action="store", default="gpu")



parser = argsparse.parse_args()
im_path = parser.im_path
checkpoint = parser.checkpoint
top_k = parser.top_k
category_names = parser.category_names
gpu = parser.gpu


if isinstance(im_path, list):
        try:
            checkpoint = im_path[1]
        except:
            pass
        im_path = im_path[0]


        
if gpu == 'False':
    gpu = False
else:
    gpu = True

    
model = ModelSpecial()    
# mapping of classes to indices which you get from  the image dataset

if gpu and torch.cuda.is_available():
        print('Using GPU')
        device = torch.device("cuda:0")
        model.cuda()
else:
    print('Using CPU')
    device = torch.device("cpu") 

model,optimizer = load_model(checkpoint)
model = model.to(device)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
prob,classes = predict(im_path, model, top_k, cat_to_name, True, device)
print('top {} probabilitys are : {}'.format(top_k, prob))
print('top {} classes are : {}'.format(top_k, classes))