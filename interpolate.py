from ast import Str
from PIL.Image import new
import h5py
import numpy
from ToolsInDevelopment.DMKSGeneration import StochasticGeneration
import matplotlib.pyplot as plt
from ToolsInDevelopment.DMKSGeneration.HelperFunctions_StochasticGeneration import twopointstats
from preprocessing import Structures
import numpy as np

'''
1. import 2 structures
2. compute their 2PS
3. interpolate
4. use StochasticGenerator to get structures
5. plot all 2PS
6. plot all structures
'''


path = '/Users/andrew/Dropbox (GaTech)/Elastic_Localization_Data/cr_50_full/31_c50_test_micros.h5'
dat = h5py.File(path, 'r')

a = dat['micros'][1459][None]
b = dat['micros'][8000][None]
x = np.concatenate((a,b), axis=0)
m = Structures(x)
ps = m.calc_statistics()


'''
phase_1 = .4
new_ps = (phase_1 * ps[0] + (1-phase_1) * ps[1])[...,None]
new_ps = np.fft.ifftshift(new_ps, axes=(0,1,2))

generator = StochasticGeneration.EigenGenerator

gen = generator(new_ps, 'incomplete') # complete is the default parameter indicating that a complete row of 
# 2PS have been given (a complete row is returned by twopointstats). 
sm1, sm2 = gen.generate()

new_ps = np.fft.fftshift(new_ps, axes=(0,1,2))

fig, ax = plt.subplots(3,2)
ax[0,0].imshow(x[0,0,:,:,15])
ax[0,0].set_title('Structure A')

ax[0,1].imshow(ps[0,:,:,15].real)
ax[0,1].set_title('2-point statistics A')

ax[1,0].imshow(x[1,0,:,:,15])
ax[1,1].imshow(ps[1,:,:,15].real)

ax[1,0].set_title('Structure B')
ax[1,1].set_title('2-point statistics B')

ax[2,0].imshow(sm2[:,:,15,0])
ax[2,1].imshow(new_ps[:,:,15,0])

ax[2,0].set_title('Generated Structure')
ax[2,1].set_title(f'{phase_1} of A 2PS + {1-phase_1} of B 2PS')

plt.show()
'''


for i in range(9):
    phase_1 = round((i+1) / 10, 1)
    phase_2 = round(1-phase_1, 1)
    new_ps = (phase_1 * ps[0] + phase_2 * ps[1])[...,None]
    new_ps = np.fft.ifftshift(new_ps, axes=(0,1,2))

    generator = StochasticGeneration.EigenGenerator

    gen = generator(new_ps, 'incomplete') # complete is the default parameter indicating that a complete row of 
    # 2PS have been given (a complete row is returned by twopointstats). 
    sm1, sm2 = gen.generate()

    new_ps = np.fft.fftshift(new_ps, axes=(0,1,2))

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(sm1[:,:,15,0])
    ax[1].imshow(new_ps[:,:,15,0])

    ax[0].set_title('Generated Structure')
    ax[1].set_title(f'{phase_1} of A 2PS + {phase_2} of B 2PS')
    plt.savefig(f'/Users/andrew/Dropbox (GaTech)/Projects/Property_Closure/Figures/{phase_1}A_{phase_2}B.png')
    plt.close