import os
import numpy
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from mocnn.MOCNN import MO_CNN
from mocnn.dataloader import LoadData
from preprocessing import *

cwd = os.getcwd()
save_dir = '/Users/andrew/Dropbox (GaTech)/ME-DboxMgmt-Kalidindi/Andrew Mann/data'

dir = '/storage/home/hhive1/amann37/scratch/homogenization_data'
#dir = os.path.join(cwd, '..', '..', '..', 'ME-DboxMgmt-Kalidindi', 'Andrew Mann', 'data')
print(dir)


device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print(f'Using {device}')
'''
profiler = pl.profiler.PyTorchProfiler(use_cuda=False, filename='prof.txt', record_shapes=True, profile_memory=True)
profiler = None
trainer = pl.Trainer(max_epochs=1, val_check_interval=1.0, progress_bar_refresh_rate=0, profiler=profiler)
'''

def main():
    print('starting up the matrix')

    seed = None
    np.random.seed(seed)

    model = MO_CNN()
    model = model.float()
    trainer = pl.Trainer(max_epochs=4, gpus=-1)

    train_data = LoadData(dir, 'train')
    valid_data = LoadData(dir, 'valid')

    train_loader = DataLoader(train_data, batch_size=32, pin_memory=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=32, pin_memory=True, num_workers=4)

    print('data loaded')

    trainer.fit(model, train_dataloaders=train_loader, valid_dataloader=valid_loader)
    trainer.save_checkpoint("models/test.ckpt")

    del train_data
    del valid_data
    del train_loader
    del valid_loader

    print('training complete!')
    print('loading test data!')


    test_data = LoadData(dir, 'test')
    test_loader = DataLoader(test_data, batch_size=32, pin_memory=True, num_workers=4)

    trainer.test(model, test_dataloaders=test_loader)

    results = model.return_results()
    results = results.cpu().numpy()

    np.save(os.path.join(dir, 'results.npy'), results)



if __name__ == "__main__":
    main()