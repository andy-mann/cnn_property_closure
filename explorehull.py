import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import h5py
import pickle
from DMKSGeneration.StochasticGeneration import *
from DMKSGeneration.HelperFunctions_StochasticGeneration import *

cwd = os.getcwd()
in_dir = os.path.join(cwd, 'inputs')

#f = open(os.path.join(in_dir, 'pca'), 'rb')
#pca = pickle.load(f)
#f.close()

scores = np.load(os.path.join(in_dir, 'scores.npy')) #8192x8192
components = np.load(os.path.join(in_dir, 'components.npy')) #8192x29791

dat = h5py.File(os.path.join(in_dir, 'train_stats.h5'), 'r')
stats = dat['2PS'] #8192x31x31x31


mean = np.mean(stats,axis=0)

def reconstruct(scores, comps, mean, dim=5, dp=0):
    scores = scores[dp,:dim][None]
    comps = comps[:dim, :].reshape(dim,31,31,31)
    return np.einsum('ds,sijk->dijk', scores,comps) + mean

def rmse(true, pred):
    mse = (true - pred)**2 / 31**3
    rmse = np.sqrt(mse)
    return rmse * 100

def ae(true,pred):
    return abs(true-pred)/np.max(true.flatten())


def figure(stats,recon, dim):
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(stats[0,:,:,15].real)

    ax[1].imshow(recon[0,:,:,15].real)

    im = ax[2].imshow(ae(stats[0].real, recon[0].real)[:,:,15])
    plt.colorbar(im)
    plt.savefig(os.path.join(cwd, 'output', f'{dim}.png'))
    plt.close()

'''
for dim in range(50):
    recon = reconstruct(scores,components, mean, dim)
    figure(stats,recon, dim)

for i in range(100):
    basis = components[i].reshape(1,31,31,31).real
    im = plt.imshow(basis[0,:,:,15])
    plt.colorbar(im)
    plt.savefig(os.path.join(cwd, 'output', 'basis', f'component_{i}.png'))
    plt.close()
'''

dim = 6
hull = ConvexHull(scores[:,:dim])


for i in range(40):
    print(i)
    corner_idx = hull.vertices[i]

    corner_stat = reconstruct(scores, components, mean, dim=dim, dp=corner_idx)
    stat = stats[corner_idx][None]

    cs = np.fft.ifftshift(stat, axes=(1,2,3))
    generator = EigenGenerator
    gen = generator(np.squeeze(cs)[...,None], 'incomplete') #input should be dsxdsxdsxn where n is the index for the covariance matrix

    sm1, sm2 = gen.generate() #output shape is dsxdsxdsxh where h is the phase

    fig, ax = plt.subplots(3)
    ax[0].imshow(corner_stat[0,:,:,15].real)
    ax[1].imshow(sm1[:,:,15,0])
    ax[2].imshow(sm2[:,:,15,0])
    plt.show()
    #plt.savefig(os.path.join(cwd, 'output', 'hull_vertices', f'index_{i}.png'))
    #plt.close()