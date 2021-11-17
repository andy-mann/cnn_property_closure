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
            ConvBlock(64,128),
	        ConvBlock(128,64),
            ConvBlock(64,32),
            ConvBlock(32,16),
            ConvBlock(16,8),
            ConvBlock(8,1,1),
            Mean()
        )

    def forward(self,x):
        return self.MultiOutputCNN(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.SimpleCNN = nn.Sequential(
            ConvBlock(1,64),
            ConvBlock(64,32),
            ConvBlock(32,16),
            ConvBlock(16,8),
            ConvBlock(8,1,1),
            Mean()
        )

    def forward(self,x):
        return self.SimpleCNN(x)
