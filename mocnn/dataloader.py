from torch.utils.data import Dataset
import numpy as np
import torch
import h5py
import os

class LoadData(Dataset):
    def __init__(self, dir, mode):
        self.dir = dir
        self.mode = mode

        self.property = self._loadProp()
        self.stats = self._loadStats()
        
        #self.micros = self._loadMicros()

    def _loadMicros(self):
        micros = h5py.File(os.path.join(self.dir, f'{self.mode}_micros.h5'), 'r')
        micros = np.asarray(micros['micros'])
        micros = micros[:,0,:,:,:] - .5
        return micros


    def _loadStats(self):
        stats = h5py.File(os.path.join(self.dir, f'{self.mode}.h5'), 'r')
        stats = np.asarray(stats['2PS']).real        
        return stats

    def _loadProp(self):
        property = h5py.File(os.path.join(self.dir, f'{self.mode}.h5'), 'r')
        property = np.asarray(property['effective_stiffness'])
        #property = (property - np.min(property,axis=0)) / (np.max(property,axis=0) - np.min(property,axis=0))
        property = (property - np.array((161.5,46.15)) / np.array((8067.9,2307)) - np.array((161.5,46.15)))
        return property

    
    def __len__(self):
        return len(self.stats)

    def __getitem__(self,idx):
        x = self.stats[idx][None]
        y = self.property[idx]
        x = torch.as_tensor(x).float()
        y = torch.as_tensor(y).float()
        return x,y
