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
def reconstruct(scores, comps, mean, idx, dim=5):
    '''
    scores
    comps
    mean
    '''
    scores = scores[idx,:dim]
    comps = comps[:dim, ...].reshape(dim,31,31,31)
    return np.einsum('s,sijk->ijk', scores,comps) + mean

def twoptcombo(x1, x2, alpha):
    '''
    x2 should be the boundary point
    '''
    return (1-alpha) * x1 + (alpha)*x2

def extrapolate(centroid, vertex, step_size=.0001):
    candidate = vertex
    alpha = 1
    count = 1
    valid = True
    while valid and count < 1000:
        print(count)
        new_point = twoptcombo(centroid, vertex, alpha)
        if check_valid(new_point):
            valid = True
            candidate = new_point
        else:
            valid = False
        alpha += step_size
        count += 1
    return candidate


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

scores = np.load(os.path.join(in_dir, 'scores.npy')) #8192x8192 - these have been mean centered and fftshifted 
components = np.load(os.path.join(in_dir, 'components.npy')) #8192x29791 - these have been mean centered and fftshifted 
components = components.reshape(8192,31,31,31)

dat = h5py.File(os.path.join(in_dir, 'train_stats.h5'), 'r')
stats = dat['2PS'] #8192x31x31x31
stats =np.fft.ifftshift(stats, axes=(1,2,3))
mean = np.mean(stats,axis=0)
print(mean[0,0,0])

dat = h5py.File(os.path.join(in_dir, 'train_stiffness.h5'), 'r')
properties = dat['effective_stiffness']

#--------------------step 2----------------#
#dim = 4
#hull = ConvexHull(scores[:,:dim])
#vertices = hull.vertices
pc_hull = ConvexHull(properties)
vertices = pc_hull.vertices


#--------------------step 3----------------#
centroid_score = np.mean(scores, axis=0)[None]
centroid = reconstruct(centroid_score, components, mean, -0)
#plt.imshow(centroid[0,:,:,15].real)
#plt.show()
for idx, vertex in enumerate(vertices):
    print('---------------------------')
    #m = reconstruct(scores, components, mean, vertex, dim=dim)
    #check_valid(m)
    #check_valid(centroid[0])
    check_valid(stats[vertex])

#--------------------step 4----------------#

print(f'there are {len(vertices)} boundary points')
boundary = np.zeros((len(vertices),10, 31,31,31))


#turns O(n) into O(n^2)...
for i in range(len(vertices)):
    for j in range(10):
        print(f'point {i}, {j}')
        boundary[i, j] = extrapolate(np.squeeze(stats[np.take(vertices, [i+1+j*2], axis=0, mode='wrap')]), stats[vertices[i]])

boundary = np.reshape(boundary,(len(vertices)*10, 31,31,31))
struct = Structures()
struct.write_data(np.fft.fftshift(boundary, axes=(1,2,3)), 'pc_interpolate')

#--------------------step 5----------------#
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

