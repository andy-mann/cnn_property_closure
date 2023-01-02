import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from DMKSGeneration.StochasticGeneration import *
from DMKSGeneration.HelperFunctions_StochasticGeneration import *
from tools import *

cwd = os.getcwd()

fp = os.path.join(cwd, 'protocol','interpolated_iter1.npy')

stats = np.load(fp)
#stats = np.fft.ifftshift(stats, axes=(1,2,3))


generator = EigenGenerator

n = len(stats)

micros = np.zeros((n,2,31,31,31))

for i in range(n):
    gen = generator(stats[i][...,None], 'incomplete')
    gen.filter('flood', alpha=0.3, beta=0.35)
    m = gen.generate(1)
    m = np.moveaxis(m,-1,0)
    micros[i] = np.squeeze(m)


files = n // 200

for i in range(files):
    write = os.path.join(cwd, 'protocol', f'interpolated_micros_{i+1}.h5')

    hf = h5py.File(write,'w')
    hf.create_dataset('micros',data=micros[i*200:i*200+200],compression='gzip')
    hf.close()