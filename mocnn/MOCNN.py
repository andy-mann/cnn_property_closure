import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from mocnn.networks import *
from torch import nn, optim

class MO_CNN(pl.LightningModule):
    def __init__(self, use_cuda=True):
        super().__init__()
        #self.save_hyperparameters()
        self.results = None

        self.net = MultiOutputCNN()
        #self.net = self.net.to(self.device)

        self.loss_fn = nn.MSELoss()

    def loss(self, pred, y):
        fxn = nn.MSELoss()
        loss = fxn(pred, y)
        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=.001)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=60, eta_min=1e-8)

        return {
            'lr_scheduler': self.scheduler,
            'optimizer': self.optimizer
        }

    def training_step(self, batch, batch_idx):
        X, y = batch
        #compute loss
        y_pred = self.forward(X)
        loss = self.loss(y_pred, y)
            
            #backprop
                #self.optimizer.zero_grad()
                #loss.backward()
                #self.optimizer.step()
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        with torch.no_grad():
            y_pred = self.forward(X).detach()
            loss = self.loss(y_pred, y).detach()

    def test_step(self,batch, batch_idx):
        X, y = batch

        with torch.no_grad():
            y_pred = self.forward(X).detach()
            loss = self.loss(y_pred, y).detach()

        if self.results is None:
            self.results = y_pred
        else:
            self.results = torch.cat((self.results, y_pred), dim=0)

    def return_results(self):
        return self.results

    def forward(self, x):
        return self.net(x)