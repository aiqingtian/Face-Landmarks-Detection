from __future__ import print_function, division
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms,utils
from utils import corrections, accuracy, acc_plot
from image_proprecessing import Rescale, RandomHorizontalFlip, RandomCrop, ToTensor


def train_mode(model, criterion, optimizer, dataset_sizes, num_epochs = 100, radius = 0.01):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dic())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('*'* 10)
        for phase in['train', 'eval']:
            if phase == 'train':
                'scheduler.step()'
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    'running loss'
                    running_loss += loss.item()* inputs.size(0)
                    running_corrects = corrections(outputs, labels, radius)
                epoch_loss = running_loss/ dataset_sizes[phase]
                print('Attention: ', dataset_sizes[phase])
                epoch_acc = running_corrects.double()/ dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                if phase == 'eval' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:0f}s'.format(
            time_elapsed //60, time_elapsed % 60))
        print('Best eval Acc: {:4f}'.format(best_acc))
        model.load_state_dict(best_model_wts)
        return model

def prediction_model(model, test_data):
    was_training = model.training
    model.eval()
    images_so_far = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data):
            inputs = inputs.to(device)
            outputs = model(inputs)
            model.train(mode = was_training)
    return  outputs


# Initialize the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torchvision.models.alexnet(pretrained = True)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 14)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
root_dir = 'lfw'
data_transforms = {
    'train':transforms.Compose([
        RandomCrop(224),
        Rescale(227),
        transforms.ColorJitter(),
        RandomHorizontalFlip(),
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        Rescale(227),
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        Rescale(227),
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
image_datasets = {x: datasets.ImageFolder(os.path.join(root_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'eval', 'test']}

# Num_workers should be set to 0 in windows machine
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= 4,
                                              shuffle= True, num_workers = 0)
               for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}



