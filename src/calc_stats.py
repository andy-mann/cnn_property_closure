from numpy.lib.npyio import save
from tools import *
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py


save_dir = '/Users/andrew/Dropbox (GaTech)/ME-DboxMgmt-Kalidindi/Andrew Mann/data'


cwd = os.getcwd()
#------------------load microstructure data---------------------#
train_m = os.path.join(cwd, '..', '..', '..', 'Elastic_Localization_Data', 'cr_50_full', '31_c50_train_micros.h5')
test_m = os.path.join(cwd, '..', '..', '..', 'Elastic_Localization_Data', 'cr_50_full', '31_c50_test_micros.h5')
valid_m = os.path.join(cwd, '..', '..', '..', 'Elastic_Localization_Data', 'cr_50_full', '31_c50_valid_micros.h5')

#train_m = h5py.File(train_m)
#test_m = h5py.File(test_m)
valid_m = h5py.File(valid_m)

#train_data = train_m['micros']
#test_data = test_m['micros']
valid_data = valid_m['micros']

#m = np.concatenate((train_data, test_data, valid_data), axis=0)

x = Structures(valid_data)
stats = x.calc_statistics()

#np.save(os.path.join(save_dir, 'pca_structures.npy'), stats)


hf = h5py.File(os.path.join(save_dir, 'valid_stats'), 'w')
hf.create_dataset('2PS', data=stats, compression='gzip')