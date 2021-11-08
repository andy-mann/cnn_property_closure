from .layers import *
from torch import nn
import torch

class MultiOutputCNN(nn.Module):
    def __init__(self):
        super(MultiOutputCNN, self).__init__()


        self.MultiOutputCNN = nn.Sequential(
            ConvBlock(1,8,2),
            ConvBlock(8,16),
            ConvBlock(16,16,2),
            ConvBlock(16,8),
            ConvBlock(8,1,1),
            Mean()
        )


    def forward(self,x):
        return self.MultiOutputCNN(x)