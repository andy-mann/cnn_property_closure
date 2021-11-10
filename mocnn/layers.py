from torch import nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3):
        super(ConvBlock, self).__init__()

        self.ks = kernel_size
        self.din = channels_in
        self.dout = channels_out


        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=self.din, out_channels=self.dout, kernel_size=self.ks, stride=1),
            nn.PReLU(num_parameters=self.dout)
	        #nn.ReLU()
        )

    def forward(self,x):
        return self.block(x)


class Mean(nn.Module):
    def __init__(self):
        super(Mean, self).__init__()

    def forward(self,x):
        return torch.mean(x,(1,2,3,4))[:,None]
