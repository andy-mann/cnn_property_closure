import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import h5py
import pickle
from DMKSGeneration.StochasticGeneration import *
from DMKSGeneration.HelperFunctions_StochasticGeneration import *
from preprocessing import *

'''
1. import PC scores (complete dimensionality)
2. compute convex hull of reduced dimensions
3. compute centroid (complete dimensionality?)
4a. function: linear combination to extrapolate
4b. function: check for valid stats
5. line bisection from valid 2PS to new* boundary
'''
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


def check_valid(candidate):
    '''
    stats must have max value at 0,0,0
    must remove idx dimension
    '''
    fstat = np.fft.fftn(candidate, axes=(0,1,2))
    if np.unravel_index(np.argmax(candidate), candidate.shape) == (0,0,0):
        print('(0,0,0) component is largest')
        if abs(np.max(candidate.imag)) < 1e-9:
            print('strictly real')
            if np.min(candidate.real) > -1e-9:
                print('stricly non-negative')
                if abs(np.max(fstat.imag)) < 1e-9 and np.min(fstat.real) > -1e-9:
                    print('fft is real and strictly non-negative')
                    if fstat.min().real > -1e-12:
                        print('This is a valid set of 2-point statistics!')
                        return True
                    else:
                        print('invalid!')
                        return False
                else:
                    print('invalid!')
                    return False
            else:
                print('invalid!')
                return False
        else:
            print('invalid!')
            return False
    else:
        print('invalid!')
        return False
                

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
struct = Structures()
struct.write_data(np.fft.fftshift(boundary, axes=(1,2,3)), 'pc_interpolate')

#--------------------step 5----------------#
'''
structures = np.zeros((10*len(vertices), 31,31,31,2))
for i in range(10*len(vertices)):
    print(f'this is boundary point {i}')
    structure = generate(boundary[i])
    structures[i] = structure
    fig, ax = plt.subplots(2)

    ax[0].imshow(np.fft.fftshift(boundary, axes=(1,2,3))[i,:,:,15].real)
    ax[1].imshow(structure[:,:,15,0])
    plt.savefig(os.path.join(cwd, 'output', 'pc_points', f'bp_{i}.png'))
    plt.close()
'''
