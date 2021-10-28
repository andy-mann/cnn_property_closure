from layers import *
from torch import nn

class MOCNN(nn.Module):
    def __init__(self):
        super(MOCNN, self).__init__()


        self.MOCNN = nn.Sequential(
            ConvBlock(1,8),
            ConvBlock(8,16),
            ConvBlock(16,16),
            ConvBlock(16,8),
            ConvBlock(8,1,1),
            nn.AvgPool3d(1)
        )


    def forward(self,x):
        return self.MOCNN(x)