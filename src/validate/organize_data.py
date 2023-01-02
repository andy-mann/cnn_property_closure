import h5py
import numpy as np
import os

cwd = os.getcwd()
base = os.path.join(cwd,'..','protocol','interior')


def calc_statistics(structures):
    f_k = np.fft.fftn(structures, axes=(2,3,4))
    F = 1/31**3 * np.conjugate(f_k[:,0,...]) * f_k[:,0,...]
    f = np.fft.ifftn(F, axes=(1,2,3))
    fshift = np.fft.fftshift(f, axes=(1,2,3))
    return fshift

m = []
stats = []
for i in range(5):
    print(f'reading micro file {i+1}')
    fpath = os.path.join(base, f'interpolated_micros_{i+1}.h5')
    dat = h5py.File(fpath)

    micros  = dat['micros']
    m.append(micros)
    stats.append(calc_statistics(micros))

m = np.concatenate(m)
stats = np.concatenate(stats)
print(m.shape)
print(stats.shape)

print('saving micros')
fpath = os.path.join(base, 'interior_gen_micros.h5')
hf = h5py.File(fpath, 'w')
hf.create_dataset('micros',data=m, compression='gzip')
hf.close()

print('saving stats')
fpath = os.path.join(base, 'interior_gen_stats.h5')
hf = h5py.File(fpath, 'w')
hf.create_dataset('2PS',data=stats, compression='gzip')
hf.close()

print('done')