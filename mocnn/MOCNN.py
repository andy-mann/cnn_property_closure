import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from mocnn.networks import *
from torch import nn, optim

class MO_CNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.results = None

        self.net = SimpleCNN()

        self.train_num_accum = 0
        self.train_loss_accum = 0
        self.val_num_accum = 0
        self.val_loss_accum = 0
        self.test_num_accum = 0
        self.test_loss_accum = 0

    def loss(self, pred, y):
        fxn = nn.L1Loss()
        loss = fxn(pred, y)
        #loss_rms = torch.sqrt(loss)
        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=.001)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=480, eta_min=1e-8)

        return {
            'lr_scheduler': self.scheduler,
            'optimizer': self.optimizer
        }

    def training_step(self, batch, batch_idx):
        X, y = batch
        #compute loss
        y_pred = self.forward(X)
        loss = self.loss(y_pred, y)

        self.train_loss_accum += loss.detach()
        self.train_num_accum += 1

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)    
        
        return loss

    def training_epoch_end(self, outputs):
        train_loss_mean = self.train_loss_accum / self.train_num_accum
        val_loss_mean = self.val_loss_accum / self.val_num_accum
        self.val_num_accum = 0
        self.val_loss_accum = 0
        self.train_num_accum = 0
        self.train_loss_accum = 0
        epoch = self.current_epoch


    def validation_step(self, batch, batch_idx):
        X, y = batch

        with torch.no_grad():
            y_pred = self.forward(X).detach()
            loss = self.loss(y_pred, y).detach()

        self.val_loss_accum += loss
        self.val_num_accum += 1

        self.log('val_loss', loss)
        return loss

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
