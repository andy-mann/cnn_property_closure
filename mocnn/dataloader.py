from torch.utils.data import Dataset
import numpy as np
import torch
import h5py
import os

class LoadData(Dataset):
    def __init__(self, dir, mode):
        self.dir = dir
        self.mode = mode

        self.stats = self._loadStats()
        self.property = self._loadProp()


    def _loadStats(self):
        stats = h5py.File(os.path.join(self.dir, f'{self.mode}_stats.h5'), 'r')
        stats = np.asarray(stats['2PS']).real        
        return stats

    def _loadProp(self):
        property = h5py.File(os.path.join(self.dir, f'{self.mode}_eff_stiffness.h5'), 'r')
        property = np.asarray(property['effective_stiffness'])
        property = (property - np.min(property)) / (np.max(property) - np.min(property))
        return property

    
    def __len__(self):
        return len(self.property)

    def __getitem__(self,idx):
        x = self.stats[idx][None]
        y = self.property[idx]

        x = torch.as_tensor(x).float()
        y = torch.as_tensor(y).float()
        return x, y