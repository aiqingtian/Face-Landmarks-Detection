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
from NN import prediction_model


def corrections(outputs, labels, radius):
    rows = labels[0]
    correct = np.ones(rows/2, 1)
    _correct = np.subtract(outputs, labels)
    _correct = _correct.reshape(rows, 2)
    for i in range(rows):
        _data= _correct[i,]
        sums = _data[0]**2 + _data[1]**2
        if sums > radius**2:
            correct[i] = 0
    return correct

def accuracy(model, test_data, labels, radius):
    outputs = prediction_model(model, test_data)
    'change radius range'
    _accuracy = []
    for i in range(10):
        radius = radius * (10**i)
        _correct = corrections(outputs, labels, radius)
        accu = _correct.sum()/ outputs.size()
        _accuracy.append([radius, accu])
    return _accuracy

def acc_plot(_accuracy):
    x = _accuracy[:, 0]
    y = _accuracy[:, 1]
    plt.plot(x, y, '-ok')