from torch import nn
import torch
from torch.nn.modules.activation import LeakyReLU


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, s=1):
        super(ConvBlock, self).__init__()

        self.ks = kernel_size
        self.din = channels_in
        self.dout = channels_out
        self.s = s


        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=self.din, out_channels=self.dout, kernel_size=self.ks, stride=s),
            nn.PReLU(num_parameters=self.dout)
	        #nn.ReLU()
            #nn.LeakyReLU()
        )

    def forward(self,x):
        return self.block(x)


class Mean(nn.Module):
    def __init__(self):
        super(Mean, self).__init__()

    def forward(self,x):
        return torch.mean(x,(2,3,4))
