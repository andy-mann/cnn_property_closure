import torch
import os
import numpy as np
from torch.nn.modules.conv import Conv3d
from torch.nn.modules.pooling import MaxPool3d
import h5py
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from preprocessing import *


cwd = os.getcwd()
save_dir = '/Users/andrew/Dropbox (GaTech)/ME-DboxMgmt-Kalidindi/Andrew Mann/data'

#dir = '/storage/home/hhive1/amann37/scratch/homogenization_data'
dir = os.path.join(cwd, '..', '..', '..', 'ME-DboxMgmt-Kalidindi', 'Andrew Mann', 'data')

dps = 32*4

#------------------load microstructure data---------------------#
train_x = os.path.join(dir, 'train_stats.h5')
test_x = os.path.join(dir, 'test_stats.h5')
valid_x = os.path.join(dir, 'valid_stats.h5')

train_x = h5py.File(train_x)
#test_x = h5py.File(test_x)
#valid_x = h5py.File(valid_x)

train_x = train_x['2PS'][:dps].real
#test_x = test_x['2PS']
#valid_x = valid_x['2PS']

#------------------load response data---------------------#

train_r = os.path.join(dir, 'train_eff_stiffness.h5')
test_r = os.path.join(dir, 'test_eff_stiffness.h5')
valid_r = os.path.join(dir, 'valid_eff_stiffness.h5')

train_r = h5py.File(train_r)
#test_r = h5py.File(test_r)
#valid_r = h5py.File(valid_r)

train_y = train_r['effective_stiffness'][:dps]
#test_y = test_r['effective_stiffness']
#valid_y = valid_r['effective_stiffness']

class LoadData(Dataset):
    def __init__(self, input, labels, transform=None, target_transform=None):
        self.input = input
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        x = self.input[idx]
        y = self.labels[idx]
        return x, y

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

train = LoadData(train_x[:,None,:,:,:], train_y)
#test = LoadData(test_x, test_y)

train_dataloader = DataLoader(train, batch_size=32)
#test_dataloader = DataLoader(test, batch_size=32)

class FlatNetwork(nn.Module):
    def __init__(self):
        super(FlatNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(31**3, 2048),
            nn.ReLU(2048),
            nn.Linear(2048, 1024),
            nn.ReLU(1024),
            nn.Linear(1024,1)
        )


    def forward(self, x):
        logits = self.network(x)
        return logits


class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv3d(1,16,3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*31**3, 512),
            nn.ReLU(),
            nn.Linear(512,1),
        )
        
    def forward(self, x):
        logits = self.network(x)
        return logits


def train(dataloader, model, loss_fx, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.float().to(device), y.float().to(device)
        print(X)
        print(y)
        #plt.imshow(X[0,0,:,:,15])
        #plt.savefig('figures')
        
        #compute loss
        pred = model(X)
        loss = loss_fx(pred, y)
        print(pred)

        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


model = ConvNetwork().to(device)
print(model)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=.001, momentum=.9)
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
print("Done!")