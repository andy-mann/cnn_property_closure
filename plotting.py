from binascii import crc32
from locale import MON_1
import matplotlib.pyplot as plt
import numpy as np
import h5py


#

# plotting 3D microstructures
dat = h5py.File('...')
micros = dat['micros']
m1 = micros[0][0]
m2 = micros[10000][0]
m3 = micros[12287][0]

c1 = np.empty(m1.shape,dtype=object)
c1[m1==0] = 'yellow'
c1[m1==1] = 'purple'

c2 = np.empty(m1.shape,dtype=object)
c2[m2==0] = 'yellow'
c2[m2==1] = 'purple'

c3 = np.empty(m1.shape,dtype=object)
c3[m3==0] = 'yellow'
c3[m3==1] = 'purple'

fig = plt.figure()

ax = fig.add_subplot(2,3,1, projection='3d')
ax.voxels(filled=m1+1, facecolors=c1)
ax.grid(False)
#ax.axis('off')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticks(ticks=[])
ax.set_yticks(ticks=[])
ax.set_zticks(ticks=[])
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_zlabel('Z')

ax = fig.add_subplot(2,3,2, projection='3d')
ax.voxels(filled=m2+1, facecolors=c2)
ax.grid(False)
ax.axis('off')

ax = fig.add_subplot(2,3,3,projection='3d')
ax.voxels(filled=m3+1, facecolors=c3)
ax.grid(False)
ax.axis('off')

ax = fig.add_subplot(2,3,4)
i1 = ax.imshow(stats[0,15,:,:].real, extent=(-15,15,-15,15))
plt.colorbar(i1, ax=ax, fraction=0.046, pad=0.04)
ax.set_xlabel('Y')
ax.set_ylabel('Z')

ax = fig.add_subplot(2,3,5)
i2 = ax.imshow(stats[10000,15,:,:].real, extent=(-15,15,-15,15))
plt.colorbar(i2, ax=ax, fraction=0.046, pad=0.04)

ax = fig.add_subplot(2,3,6)
i3 = ax.imshow(stats[12287,15,:,:].real, extent=(-15,15,-15,15))
plt.colorbar(i3, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()



colors = np.empty(data.shape, dtype=object)
colors[data==0] = 'yellow'
colors[data==1] = 'purple'

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(filled=data+1, facecolors=colors)
ax.grid(False)
ax.axis('off')

plt.show()
