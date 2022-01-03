import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import h5py
import pickle
from DMKSGeneration.StochasticGeneration import *
from DMKSGeneration.HelperFunctions_StochasticGeneration import *
from tools import *

#--------------------function definitions----------------#


def twoptcombo(x1, x2, alpha):
    '''
    x2 should be the boundary point
    '''
    return (1-alpha) * x1 + (alpha)*x2

def interpolate(pt1, pt2, step_size=.1):
    points = []
    for i in range(10):
        points.append(twoptcombo(pt1, pt2, step_size*i))
    print(np.array(points).shape)
    return np.array(points)

def calc_statistics(structure):
    f_k = np.fft.fftn(structure, axes=(2,3,4))
    print(f_k.shape)
    F = 1/31**3 * np.conjugate(f_k[:,0,...]) * f_k[:,0,...]
    print(F.shape)
    f = np.fft.ifftn(F, axes=(1,2,3))
    print(f.shape)
    #fshift = np.fft.fftshift(f, axes=(1,2,3))
    return f

def generate(stats):
    #shift_stats = np.fft.ifftshift(stats, axes=(0,1,2))[...,None]
    generator = EigenGenerator
    gen = generator(stats[...,None], 'incomplete') #input should be dsxdsxdsxn where n is the index for the covariance matrix
    return gen.generate(1)


#--------------------step 1 (import everything)----------------#
cwd = os.getcwd()
in_dir = os.path.join(cwd, 'inputs')

dat = h5py.File(os.path.join(in_dir, 'train_stats.h5'), 'r')
stats = dat['2PS'] #8192x31x31x31
stats =np.fft.ifftshift(stats, axes=(1,2,3))

dat = h5py.File(os.path.join(in_dir, 'train_stiffness.h5'), 'r')
properties = dat['effective_stiffness']

#--------------------step 2----------------#
pc_hull = ConvexHull(properties)
vertices = pc_hull.vertices

#--------------------step 3----------------#
print(f'there are {len(vertices)} boundary points')
boundary = np.zeros((len(vertices),10, 31,31,31))

for b in range(len(vertices)-1):
    boundary[b] = interpolate(stats[vertices[b]], stats[vertices[b+1]])


boundary = np.reshape(boundary,(len(vertices)*10, 31,31,31))
#struct = Structures()
#struct.write_data(np.fft.fftshift(boundary, axes=(1,2,3)), 'pc_interpolate')

#--------------------step 5----------------#

structures = np.zeros((10*len(vertices), 31,31,31,2))
for i in range(10*len(vertices)):
    print(f'this is boundary point {i}')
    structure = generate(boundary[i])
    structures[i] = structure
    structure = np.reshape(structure, (2,31,31,31))[None]
    print(structure.shape)
    stats = calc_statistics(structure)
    print(stats.shape)
    print(np.max(abs(boundary[i] - np.squeeze(stats))))

    #fig, ax = plt.subplots(2)

    #ax[0].imshow(np.fft.fftshift(boundary, axes=(1,2,3))[i,:,:,15].real)
    #ax[1].imshow(structure[:,:,15,0])
    #plt.savefig(os.path.join(cwd, 'output', 'pc_points', f'bp_{i}.png'))
    #plt.close()

