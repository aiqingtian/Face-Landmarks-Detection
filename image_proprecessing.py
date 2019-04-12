from __future__ import print_function, division
import os
import torch
import random
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils

print(os.getcwd())
landmarks_frame = pd.read_csv('lfw/LFW_annotation_train.txt')
n = 7
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

class LandmarksDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform= None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, idx):
        'Ext Images'
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float')
        landmarks_2d = landmarks.reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks, 'landmarks_2d': landmarks_2d}

        if self.transform:
            sample = self.transform(sample)
        return sample

# Transforms
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, landmarks_2d = sample['image'], sample['landmarks'], sample['landmarks_2d']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = transforms.resize(image, (new_h, new_w))
        landmarks_2d = landmarks_2d* [new_w/ w, new_h/ h]
        landmarks = landmarks_2d.reshape[-1 ,1]

        return {'image': img, 'landmarks': landmarks, 'landmarks_2d': landmarks_2d}


class RandomHorizontalFlip(object):

    def __call__(self, sample):
        image, landmarks_2d = sample['image'], sample['landmarks_2d']
        h, w = image.shape[:2]
        radn_num = random.uniform(0, 10)
        if radn_num > 5:
            image = np.flip(image, 1)
            landmarks_2d[0] = w- landmarks[0] -1
        return image, landmarks_2d

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks_2d = sample['image'], sample['landmarks_2d']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks_2d = landmarks_2d - [left, top]

        return {'image': image, 'landmarks_2d': landmarks_2d}

class ToTensor(object):
    def __call__(self, sample):
        image, landmarks_2d = sample['image'], sample['landmarks_2d']
        image = image.transpose((2, 0, 1))
        return{'image': torch.from_numpy(image),'landmarks_2d': torch.from_numpy(landmarks_2d)}
