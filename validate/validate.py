import h5py
import numpy as np
import os
import matplotlib.pyplot as plt


cwd = os.getcwd()
base = os.path.join(cwd, 'inputs', '11_responses')
arr = np.zeros((240,1))
for i in range(240):
    print(f'reading response file {i+1}')
    fpath = os.path.join(base, f'{i+1}_micros_responses.h5')
    dat = h5py.File(fpath)

    stress  = dat['stress']
    strain = dat['strain']

    #stress = np.reshape(stress, (10,6,31**3))
    #strain = np.reshape(strain, (10,6,31**3))

    stress_avg = np.sum(stress, axis=(2,3,4))
    strain_avg = np.sum(strain, axis=(2,3,4))

    s = stress_avg[:,0]
    e = strain_avg[:,0]

    eff_p = (s / e)[:,None]

    #arr[i] = np.mean(eff_p)

print('done')
np.save('val_eff_11.npy', arr)
print(arr)