import h5py
import numpy as np
import os

cwd = os.getcwd()
fp = os.path.join('/Users/andrew/Dropbox (GaTech)/2PS_Property_Data')

c1111 = [4984, 7100]
c1212 = [1071,1506]
iteration =1


dat = h5py.File(os.path.join(fp,'train.h5'))
prop = dat['stiffness']

stats = dat['2PS']
stats = np.array(stats)


a = prop[:,0]
b = prop[:,1]
c1111_slice = np.logical_and(a>c1111[0], a<c1111[1])
c1212_slice = np.logical_and(b>c1212[0], b<c1212[1])
idx = np.logical_and(c1111_slice,c1212_slice)
p1 = a[idx]
p2= b[idx]
prop_slice = np.stack((p1,p2),1)

stats_slice = stats[idx]

save_fp = os.path.join(cwd,'protocal')
hf = h5py.File(os.path.join(save_fp, f'iteration_{iteration}_stats_b.h5'), 'w')
hf.create_dataset('2PS',data=stats_slice,compression='gzip')
hf.close()

np.save(os.path.join(save_fp, f'iteration_{iteration}_stiff_b.npy'),prop_slice)

