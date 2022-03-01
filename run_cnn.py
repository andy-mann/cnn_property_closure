import os
import numpy
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from mocnn.MOCNN import MO_CNN
from mocnn.dataloader import LoadData
from mocnn.helpers import *
from mocnn.figures import *
from tools import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

cwd = os.getcwd()
save_dir = '/Users/andrew/Dropbox (GaTech)/ME-DboxMgmt-Kalidindi/Andrew Mann/data'

dir = '/storage/home/hhive1/amann37/scratch/homogenization_data'
#dir = os.path.join(cwd, '..', '..', '..', 'ME-DboxMgmt-Kalidindi', 'Andrew Mann', 'data')
print(dir)

model_indicator = 'A_updated'

test_set = 'boundary'
train = True
test = True
expand_boundary = False

def main():
    print('starting up the matrix')

    seed = None
    np.random.seed(seed)

    if train:
        model = MO_CNN()
    else:
        model = MO_CNN.load_from_checkpoint(checkpoint_path=f'/Users/andrew/Dropbox (GaTech)/code/class/materials_informatics/models/{model_indicator}.ckpt')

    model = model.float()

    trainer = pl.Trainer(max_epochs=480, gpus=-1, progress_bar_refresh_rate=0)
    #trainer = pl.Trainer(max_epochs=1)    

    if train:
        network_size = count_parameters(model)
        print(f'There are {network_size} tunable parameters in this model')

        train_data = LoadData(dir, 'new_train')
        valid_data = LoadData(dir, 'valid')

        train_loader = DataLoader(train_data, batch_size=32, pin_memory=True, num_workers=4)
        valid_loader = DataLoader(valid_data, batch_size=32, pin_memory=True, num_workers=4)

        print('data loaded')

        #trainer.fit(model, train_dataloaders=train_loader)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        trainer.save_checkpoint(f"models/{model_indicator}.ckpt")

        del train_data
        del valid_data
        del train_loader
        del valid_loader

        print('training complete!')
        print('loading test data!')

        test_data = LoadData(dir, 'test')
        test_loader = DataLoader(test_data, batch_size=32, pin_memory=True, num_workers=4)

        trainer.test(model, dataloaders=test_loader)

        predictions = model.return_results()
        predictions = predictions.cpu().numpy()

        np.save(os.path.join(os.getcwd(), 'output', f'{model_indicator}_predictions.npy'), predictions)

        x, y_test = dataset_to_np(test_data)

        MASE = mase(predictions, y_test)
        MAE = mae(predictions, y_test)

        print(f'MASE is {MASE * 100} and MAE is {MAE * 100}')

        #parity(predictions[:,0], y_test[:,0], 'C11', model_indicator, os.getcwd())
        #parity(predictions[:,1], y_test[:,1], 'C66', model_indicator, os.getcwd())
    elif expand_boundary:
        print('expanding boundary')
        #test_data = LoadData(dir, 'boundary')
        #test_loader = DataLoader(test_data, batch_size=32, pin_memory=True, num_workers=4)
        #fp = os.path.join(cwd, 'inputs', '51_51_51/truncated_51_stats.h5')
        #dat = h5py.File(os.path.join(dir, 'test_stats_u.h5'))
        #x = np.array(dat['2PS'])
        
        #fp = os.path.join(cwd, 'inputs', 'expand', 'valid.npy')
        fp = os.path.join(cwd,'inputs', 'boundary_interpolate_stats.npy')
        x = np.load(fp)
        x = x[:,None,...].real
        print(x.shape)
        x = torch.as_tensor(x).float()

        model.eval()
        predictions = model(x)
        predictions = predictions.detach().numpy()
        predictions = un_normalize(predictions, np.array(((8067.9,2307), (161.5,46.15))))
        print(predictions)

        np.save(os.path.join(os.getcwd(), 'protocal', f'{model_indicator}_boundary_cnn.npy'), predictions)
        #np.save(os.path.join(os.getcwd(), 'output', 'results', f'{model_indicator}_2PS_generated_pred_2.npy'), predictions)
    
    return model



if __name__ == "__main__":
    main()
