import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from DMKSGeneration.StochasticGeneration import *
from DMKSGeneration.HelperFunctions_StochasticGeneration import *
from tools import *

cwd = os.getcwd()

dat = h5py.File(os.path.join(cwd, 'inputs', 'pc_interpolate_stats.h5'), 'r')
boundary = dat['2PS']
boundary = np.fft.ifftshift(boundary, axes=(1,2,3))

def generate(stats):
    #shift_stats = np.fft.ifftshift(stats, axes=(0,1,2))[...,None]
    generator = EigenGenerator
    gen = generator(stats[...,None], 'incomplete') #input should be dsxdsxdsxn where n is the index for the covariance matrix
    return gen.generate(1)

def write_data(structures, case, overwrite=False):
    fp = os.path.join(os.getcwd(), 'inputs', 'micros', f'{case}_micros.h5')
    if os.path.exists(fp) and not(overwrite):
        dat = h5py.File(fp, 'r')
        structures = dat['micros']
    else:
        hf = h5py.File(fp, 'w')
        hf.create_dataset('micros', data=structures, compression='gzip')
        hf.close()


for i in range(len(boundary)):
    print(f'this is boundary point {i}')
    structures = np.zeros((10, 31,31,31,2))
    for j in range(10):
        s = generate(boundary[i])
        structures[j] = s

    write_data(structures, i+1)