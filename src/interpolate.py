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


def twoptcombo(x1, x2, alpha):
    return (1-alpha) * x1 + (alpha)*x2

def interpolate(pt1, pt2, step_size=.1):
    points = []
    for i in range(2):
        points.append(twoptcombo(pt1, pt2, (i+1)*step_size))
    return np.array(points)



cwd = os.getcwd()
fp = os.path.join(cwd,'protocol', 'iteration_1_stats.npy')
stats = np.load(fp)
print('loaded statistics')

n = len(stats)
print(f'There are {n} sets of spatial stats')


p = []
for i in range(n-1):
        p.append(interpolate(stats[i], stats[i+1], 1/3))
p = np.array(p)
p = np.reshape(p,((n-1)*2,31,31,31))

print(f'interpolated {(n-1)*2} new structures')

fp = os.path.join(cwd, 'protocol','interpolated_iter1.npy')
np.save(fp, p)

'''

 #-------------------------------------------------------------------#
cwd = os.getcwd()
in_dir = os.path.join(cwd, 'inputs')

dat = h5py.File('/Users/andrew/Dropbox (GaTech)/ME-DboxMgmt-Kalidindi/Andrew Mann/data/train_stats_u.h5', 'r')
stats = dat['2PS'] #8192x31x31x31
#stats =np.fft.ifftshift(stats, axes=(1,2,3))

dat = h5py.File('/Users/andrew/Dropbox (GaTech)/ME-DboxMgmt-Kalidindi/Andrew Mann/data/train_stiffness_u.h5', 'r')
properties = dat['effective_stiffness']

pc_hull = ConvexHull(properties)
vertices = pc_hull.vertices
np.save('vertices.npy', vertices)

print(f'there are {len(vertices)} boundary points')
boundary = np.zeros((len(vertices)*10, 31,31,31))
idx = 0
for b in range(len(vertices)-1):
    boundary[idx*10:(idx+1)*10] = interpolate(stats[vertices[b]], stats[vertices[b+1]])
    idx = idx + 1

fp = '/Users/andrew/Dropbox (GaTech)/code/class/materials_informatics/inputs/expand/boundary_interp_stats.npy'
np.save(fp, boundary)
'''