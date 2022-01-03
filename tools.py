import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

class Structures():
    def __init__(self):
        self

    def print_structure(self, idx):
        plt.imshow(self.structures[idx,0,:,:,15])
        plt.show()

    def get_vf(self):
        vf = np.sum(self.structures, axis=(2,3,4)) / 31**3
        return vf

    def plot_vf(self, phase=0):
        #plt.scatter(np.arange(len(train_data))+1, self.get_vf()[:,phase])
        #plt.show()
        plt.hist(self.get_vf()[:,phase], bins=len(self.structures)//100)
        plt.show()

    def calc_statistics(self):
        f_k = np.fft.rfftn(self.structures, axes=(2,3,4))
        F = 1/31**3 * np.conjugate(f_k[:,0,...]) * f_k[:,0,...]
        f = np.fft.irfftn(F, axes=(1,2,3))
        fshift = np.fft.fftshift(f, axes=(1,2,3))
        return fshift

    def write_data(self, stats, case, overwrite=False):
        fp = os.path.join(os.getcwd(), 'inputs', f'{case}_stats.h5')
        if os.path.exists(fp) and not(overwrite):
            dat = h5py.File(fp, 'r')
            stats = dat['2PS']
        else:
            hf = h5py.File(fp, 'w')
            hf.create_dataset('2PS', data=stats, compression='gzip')
            hf.close()




class Responses():
    def __init__(self, stress, strain):
        self.stress = stress
        self.strain = strain

    def get_effective_property(self, case, component, overwrite=False):
        fp = f'/Users/andrew/Dropbox (GaTech)/ME-DboxMgmt-Kalidindi/Andrew Mann/data/{case}_eff_stiffness.h5'

        if os.path.exists(fp) and not(overwrite):
            dat = h5py.File(fp, 'r')
            eff_p = dat['effective_stiffness']
        else:
            stress_avg = np.sum(self.stress,axis=1) / len(self.stress[0,:,0])
            strain_avg = np.sum(self.strain,axis=1) / len(self.strain[0,:,0])

            s = stress_avg[:,component]
            e = strain_avg[:,component]

            eff_p = (s / e)[:,None]
            hf = h5py.File(fp, 'w')
            print(eff_p.shape)
            hf.create_dataset('effective_stiffness', data=eff_p, compression='gzip')
            hf.close()


        return eff_p

    def plot_effective_property(self):
        array = self.get_effective_property()

        plt.scatter(np.arange(len(array[:,0])), array[:,0])
        plt.show()