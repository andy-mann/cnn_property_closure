import os
import numpy as np
import h5py
from scipy.spatial import ConvexHull
from DMKSGeneration.StochasticGeneration import *
from DMKSGeneration.HelperFunctions_StochasticGeneration import *
import matplotlib.pyplot as plt


def generate(stats):
    #shift_stats = np.fft.ifftshift(stats, axes=(0,1,2))[...,None]
    generator = EigenGenerator
    gen = generator(stats[...,None], 'incomplete') #input should be dsxdsxdsxn where n is the index for the covariance matrix
    return gen.generate(1)

def calc_statistics(structures):
    f_k = np.fft.fftn(structures, axes=(2,3,4))
    F = 1/31**3 * np.conjugate(f_k[:,0,...]) * f_k[:,0,...]
    f = np.fft.ifftn(F, axes=(1,2,3))
    #fshift = np.fft.fftshift(f, axes=(1,2,3))
    return f

def rmse(true, pred):
    N = 31**3
    se = np.sum((true - pred)**2, axis=(0,1,2))
    mse = se / N
    return np.math.sqrt(mse)

cwd = os.getcwd()
in_dir = os.path.join(cwd, 'inputs')

dat = h5py.File(os.path.join(in_dir, 'train_stats.h5'), 'r')
stats = dat['2PS'] #8192x31x31x31

mf = os.path.join('/Users/andrew/Dropbox (GaTech)/Elastic_Localization_Data/cr_50_full/31_c50_train_micros.h5')
dat = h5py.File(mf)
micros = dat['micros']

stats =np.fft.ifftshift(stats, axes=(1,2,3))
dat = h5py.File(os.path.join(in_dir, 'train_stiffness.h5'), 'r')
properties = dat['effective_stiffness']


pc_hull = ConvexHull(properties)
vertices = pc_hull.vertices

'''
for i in range(len(vertices)):
    m = micros[vertices[i]]
    stat = stats[vertices[i]]

    gm = generate(stat)
    print(m.shape)
    print(stat.shape)
    print(gm.shape)
    stat = np.fft.fftshift(stat, axes=(0,1,2))

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(m[0,:,:,15])
    ax[1].imshow(gm[:,:,15,0])
    ax[2].imshow(stat[:,:,15].real)
    plt.show()
'''

for i in range(7):
    m = micros[vertices[5]]
    stat = stats[vertices[5]]
    gm = generate(stat)
    stat = np.fft.fftshift(stat, axes=(0,1,2))

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(m[0,:,:,i*5])
    ax[1].imshow(gm[:,:,i*5,0])
    ax[2].imshow(stat[:,:,i*5].real)
    plt.show()
