import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Structures():
    def __init__(self, structures):
        self.structures = structures

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


        return eff_p

    def plot_effective_property(self):
        array = self.get_effective_property()

        plt.scatter(np.arange(len(array[:,0])), array[:,0])
        plt.show()


'''
save_dir = '/Users/andrew/Dropbox (GaTech)/ME-DboxMgmt-Kalidindi/Andrew Mann/data'


cwd = os.getcwd()
#------------------load microstructure data---------------------#
train_m = os.path.join(cwd, '..', '..', '..', 'Elastic_Localization_Data', 'cr_50_full', '31_c50_train_micros.h5')
test_m = os.path.join(cwd, '..', '..', '..', 'Elastic_Localization_Data', 'cr_50_full', '31_c50_test_micros.h5')
valid_m = os.path.join(cwd, '..', '..', '..', 'Elastic_Localization_Data', 'cr_50_full', '31_c50_valid_micros.h5')

train_m = h5py.File(train_m)
#test_m = h5py.File(test_m)
#valid_m = h5py.File(valid_m)

train_data = train_m['micros']
#test_data = test_m['micros']
#valid_data = valid_m['micros']

#m = np.concatenate((train_data, test_data, valid_data), axis=0)

#------------------load response data---------------------#

train_r = os.path.join(cwd, '..', '..', '..', 'Elastic_Localization_Data', 'cr_50_full', '31_c50_train_responses.h5')
test_r = os.path.join(cwd, '..', '..', '..', 'Elastic_Localization_Data', 'cr_50_full', '31_c50_test_responses.h5')
valid_r = os.path.join(cwd, '..', '..', '..', 'Elastic_Localization_Data', 'cr_50_full', '31_c50_valid_responses.h5')

train_r = h5py.File(train_r)
test_r = h5py.File(test_r)
valid_r = h5py.File(valid_r)

print('data loaded')

train_stress = train_r['stress']
#test_stress = test_r['stress']
#valid_stress = valid_r['stress']

train_strain = train_r['strain']
#test_strain = test_r['strain']
#valid_strain = valid_r['strain']

#e = np.concatenate((train_strain, test_strain, valid_strain), axis=0)
#s = np.concatenate((train_stress, test_stress, valid_stress), axis=0)



x = Structures(train_data[0])
#vf = x.get_vf()[:,0]
stats = x.calc_statistics()


train = Responses(train_stress, train_strain)
eff_p = train.get_effective_property(case='train')

#test = Responses(test_stress, test_strain)
#test.get_effective_property(case='test')

#valid = Responses(valid_stress, valid_strain)
#valid.get_effective_property(case='valid')
#y.plot_effective_property()
'''