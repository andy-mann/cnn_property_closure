from mimetypes import init
from .layers import *
from torch import nn
import torch

class MultiOutputCNN(nn.Module):
    def __init__(self):
        super(MultiOutputCNN, self).__init__()


        self.MultiOutputCNN = nn.Sequential(
            ConvBlock(1,8,5),
            ConvBlock(8,16),
            ConvBlock(16,32),
            ConvBlock(32,64),
            ConvBlock(64,64),  
            ConvBlock(64,32),
            ConvBlock(32,16),
            ConvBlock(16,8),
            ConvBlock(8,2,1),
            Mean()
        )

    def forward(self,x):
        return self.MultiOutputCNN(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.SimpleCNN = nn.Sequential(
            ConvBlock(1,8,5,2),
            ConvBlock(8,16),
            ConvBlock(16,32),
            ConvBlock(32,64),
            ConvBlock(64,2,1),
            Mean()
        )

        self.fc1 = nn.Linear(dim,dim)

    def forward(self,x):
        stats, cr = x[:,:d-1], x[:,d-1:]
        cnn_o = self.SimpleCNN(stats)
        x = torch.cat((cnn_o, cr), 1)
        out = self.fc1(x)
        return self.SimpleCNN(x)

class NetworkF(nn.Module):
    def __init__(self):
        super(NetworkF, self).__init__()

        self.NetworkF = nn.Sequential(
            ConvBlock(1,64,5,2),
            ConvBlock(64,32,3),
            ConvBlock(32,16,3),
            ConvBlock(16,8,3),
            ConvBlock(8,4,3),
            ConvBlock(4,2,6)
        )

    def forward(self,x):
        x = self.NetworkF(x)
        x = torch.squeeze(x)
        return x