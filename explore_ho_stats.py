import h5py
import numpy as np
from numpy.fft import fft, ifft, fftshift
import os
import matplotlib.pyplot as plt

def primitive(m):
    arr = np.zeros((5,2))

    for i in range(5):
        if m[:,i] == 1:
            arr[i,0] = 1
        else:
            arr[i,1] = 1
    return arr

def correlations(structure):
    mk = fft(structure, axis=1)

    F = 1/structure.shape[-1] * np.conjugate(mk) * mk
    f = ifft(F, axis=1)

    fshifted = fftshift(f)
    return fshifted

def tps(structure, h=2):
    mk = fft(structure, axis=0)
    arr =np.zeros(5,5,2,2,2)

    for i in range(h):
        for j in range(h):
            for k in range(h):
                for r in range(5):
                    arr[:,r,i,j,k] = 1/structure.shape[0] * np.conjugate(mk[:,i]) * mk[-r, j] * mk[r,k]


def new_structures(dim):
    structure = np.zeros((dim))
    for i in range(structure.shape[-1]):
        u = np.random.rand()
        if u <= .5:
            structure[:,i] = 1
        else:
            pass
    return structure