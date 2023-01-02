import os
import numpy as np
import h5py

def calc_statistics(structures):
        f_k = np.fft.fftn(structures, axes=(2,3,4))
        F = 1/31**3 * np.conjugate(f_k[:,0,...]) * f_k[:,0,...]
        f = np.fft.ifftn(F, axes=(1,2,3))
        #fshift = np.fft.fftshift(f, axes=(1,2,3))
        return f

def rmse(true, pred):
    N = 31**3
    se = np.sum((true - pred)**2, axis=(0,1,2))
    mse = se / N
    return np.math.sqrt(mse)

cwd = os.getcwd()

fpath = os.path.join(cwd, '..', 'inputs', 'micros')

d = h5py.File(os.path.join(cwd, '..', 'inputs', 'boundary_interpolate_stats.h5'))
b_stats = d['2PS']

save = np.zeros((240,31,31,31))
for i in range(240):
    fp = os.path.join(fpath, f'{i+1}_micros.h5')
    dat = h5py.File(fp)
    m = dat['micros']
#    m = np.reshape(m, (10,2,31,31,31))
    m = np.moveaxis(m, 4, 1)
    print(m.shape)

    stats = calc_statistics(m)
    avg_stat = np.mean(stats,axis=0)
    save[i] = avg_stat

    #for j in range(10):
    #    print(b_stats[i].shape)
    #    print(stats[j].shape)
    #    error = rmse(b_stats[i], stats[j])
    #    print(error)

np.save('avg_stats.npy', save)
