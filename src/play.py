from DMKSGeneration.StochasticGeneration import *
from DMKSGeneration.HelperFunctions_StochasticGeneration import *

import numpy as np


def calc_statistics(structures):
     f_k = np.fft.fftn(structures, axes=(2,3,4))
     F = 1/31**3 * np.conjugate(f_k[:,0,...]) * f_k[:,0,...]
     f = np.fft.ifftn(F, axes=(1,2,3))
     return f

BP = np.load('../char_shift/cs.npy')
#BP = np.fft.ifftshift(BP,axes=(1,2,3))

generator = EigenGenerator
gen = generator(BP[0][...,None], 'incomplete')
m = gen.generate(1)
m = np.moveaxis(m,-1,0)

stat = calc_statistics(m[None])

#i want to store the statistics...
stats = np.zeros((100,31,31,31))
stats[0] = np.squeeze(stat)

micros[0] = np.squeeze(m)

for i in range(100):
    gen = generator(stat[0][...,None], 'incomplete')
    m = gen.generate(1)
    m = np.moveaxis(m,-1,0)
    stats[i+1] = calc_statistics(m[None])
    micros[i+1] = np.squeeze(m)