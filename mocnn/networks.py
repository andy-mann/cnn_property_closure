from .layers import *
from torch import nn
import torch

class MOCNN(nn.Module):
    def __init__(self):
        super(MOCNN, self).__init__()


        self.MOCNN = nn.Sequential(
            ConvBlock(1,8,2),
            nn.MaxPool3d(2,2),
            ConvBlock(8,16),
            ConvBlock(16,16,2),
            nn.MaxPool3d(2,2),
            ConvBlock(16,8),
            ConvBlock(8,1,1),
            #nn.AvgPool3d(4),
            Mean()
        )


    def forward(self,x):
        return self.MOCNN(x)