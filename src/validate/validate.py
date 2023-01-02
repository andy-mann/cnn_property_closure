import h5py
import numpy as np
import os
import matplotlib.pyplot as plt


cwd = os.getcwd()
base = os.path.join(cwd, '..', 'protocol', 'interior')
eff_xx = []
eff_xy = []
for i in range(5):
    print(f'reading response file {i+1}')
    fpath = os.path.join(base, f'interpolated_micros_{i+1}_responses_xx.h5')
    dat = h5py.File(fpath)

    stress  = dat['stress']
    strain = dat['strain']

    #stress = np.reshape(stress, (10,6,31**3))
    #strain = np.reshape(strain, (10,6,31**3))

    stress_avg = np.sum(stress, axis=(2,3,4))
    strain_avg = np.sum(strain, axis=(2,3,4))

    s = stress_avg[:,0]
    e = strain_avg[:,0]

    arr = (s / e)[:,None]
    eff_xx = np.append(eff_xx,np.squeeze(arr),0)



for i in range(5):
    print(f'reading response file {i+1}')
    fpath = os.path.join(base, f'interpolated_micros_{i+1}_responses_xy.h5')
    dat = h5py.File(fpath)

    stress  = dat['stress']
    strain = dat['strain']

    #stress = np.reshape(stress, (10,6,31**3))
    #strain = np.reshape(strain, (10,6,31**3))

    stress_avg = np.sum(stress, axis=(2,3,4))
    strain_avg = np.sum(strain, axis=(2,3,4))

    s = stress_avg[:,3]
    e = strain_avg[:,3]

    arr = (s / e)[:,None]
    eff_xy = np.append(eff_xy,np.squeeze(arr),0)


eff_p = np.stack((np.squeeze(eff_xx), np.squeeze(eff_xy)), 1)

print('done')
fp = os.path.join(cwd,'..', 'protocol','interior','interior_fea.npy')
np.save(fp, eff_p)