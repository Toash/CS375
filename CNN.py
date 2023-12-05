from IPython.display import Math
from IPython.display import Latex

import numpy as np
import os
import struct
from scipy import datasets

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class_labels = {
        "ant": "./Data/full_numpy_bitmap_ant.npy",
        "bucket": "./Data/full_numpy_bitmap_bucket.npy",
        "cow": "./Data/full_numpy_bitmap_cow.npy",
        "crab": "./Data/full_numpy_bitmap_crab.npy",
        "dragon": "./Data/full_numpy_bitmap_dragon.npy",
        "fork": "./Data/full_numpy_bitmap_fork.npy",
        "lollipop": "./Data/full_numpy_bitmap_lollipop.npy",
        "moon": "./Data/full_numpy_bitmap_moon.npy",
        "pizza": "./Data/full_numpy_bitmap_pizza.npy",
        "zigzag": "./Data/full_numpy_bitmap_zigzag.npy",
}

def get_output_label(inputs):
    """
    Return the most likely class label of the input. 
    Args:
        input (tensor): input 28 x 28 image 
    """
    return list(class_labels.keys())[torch.argmax(inputs, dim=1)]

#   CNN model
class Net(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QuickDrawDataset(Dataset):
    """
    Class for loading google dataset. Based on
    https://discuss.pytorch.org/t/input-numpy-ndarray-instead-of-images-in-a-cnn/18797.
    """
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.data)
    
def load_dataset(batch_size=10, classes=2):
    """
    There is a LOT of data. For now, we only load ~2.5% (10,000 data points) 
    of each label for training speed.
    Returns:
        a tuple containing the training and testing dataset and loader
        containing the selected labels from the google quickdraw dataset.
    """
    X = np.empty((0, 784))
    y = np.empty((0, 1))

    class_label = 0
    for _, file in class_labels.items():
        dataset = np.load(file)
        #   Get 10,000 samples of each label
        X = np.concatenate((X, dataset[:10000]), axis=0)
        y = np.concatenate((y, np.full((10000, 1), class_label)), axis=0)
        
        #   Each class gets a unique value for their label
        class_label += 1

        #   Stop running if the desired number of classes is reached
        if class_label >= classes:
            break

    #   Normalizing X
    X = ((X / 255.0) - 0.5) * 2
    X = np.reshape(X, (X.shape[0], 1, 28, 28))  #   1 color channel for 28x28 image.
    y = np.reshape(y, (y.shape[0], ))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=123, stratify=y
    )

    transform = transforms.Compose([
        transforms.RandomAffine(degrees=30, translate=(.2, .2), scale=(.8, 1.2)),
        transforms.RandomCrop(28, pad_if_needed=True)
    ])

    trainset = QuickDrawDataset(X_train, y_train, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = QuickDrawDataset(X_test, y_test, transform=transform)
    testoader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return (trainset, trainloader, testset, testoader)

def train_model(file_path="./model.pth", max_iterations=10, batch_size=4, classes=2):
    """
    Trains a CNN model on selected labels from the google quick draw dataset.
    """
    torch.manual_seed(0)

    _trainset, trainloader, _testset, testloader = load_dataset(batch_size, classes)

    try:
        net = Net(classes)
        net.load_state_dict(torch.load(file_path))
        return net
    except:
        print("Couldn't load model from file:", file_path)
    net = Net(classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #   Statistics
    trainingLoss = []
    trainingAccuracy = []
    testLoss = []
    testAccuracy = []

    #   Training model
    for epoch in range(1, max_iterations+1):  # loop over the dataset multiple times
        running_loss = 0.0
        training_loss = 0.0
        accuracy = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            training_loss += loss.item()
            accuracy += sum(torch.argmax(outputs, dim=1) == labels).item() / float(batch_size)
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        trainingLoss.append(training_loss / len(trainloader))
        trainingAccuracy.append(accuracy / len(trainloader))
        print("Epoch:", epoch, "loss:", trainingLoss[-1])
        print("Epoch:", epoch, "accuracy:", trainingAccuracy[-1])
    print(trainingLoss)
    print(trainingAccuracy)

    torch.save(net.state_dict(), file_path)
    return net