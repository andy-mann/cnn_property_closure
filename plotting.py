import matplotlib.pyplot as plt
import numpy as np
import h5py


#

# plotting 3D microstructures
dat = h5py.File('...')
micro = dat['micro']
data = micro[0][0]
colors = np.empty(data.shape, dtype=object)
colors[data==0] = 'yellow'
colors[data==1] = 'purple'

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(filled=data+1, facecolors=colors)

