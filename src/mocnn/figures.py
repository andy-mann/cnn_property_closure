import matplotlib.pyplot as plt
import numpy as np
import os
#from mocnn.helpers import *

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 250
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 
plt.rcParams.update({'font.size':8})


def parity(p_train, t_train, p_test, t_test):
    fig, ax = plt.subplots(1,2)

    ax[0].plot([0, 8067], [0, 8067], 'k-', lw=2)
    ax[0].plot(t_train[:,0], p_train[:,0], 'o', c='lightgrey', alpha=.7,label='Train')
    ax[0].plot(t_test[:,0], p_test[:,0], 'o', c='r', alpha=.1,label='Test')
    ax[0].xlabel(r'Target $\displaystyle C_{1111}$')
    ax[0].ylabel(r'Prediction $\displaystyle C_{1111}$')
    ax[0].legend()
    ax[0].xlim(left=0)
    ax[0].ylim(bottom=0)

    ax[1].plot([0, 2307], [0, 2307], 'k-', lw=2)
    ax[1].plot(t_train[:,1], p_train[:,1], 'o', c='lightgrey', alpha=.7,label='Train')
    ax[1].plot(t_test[:,1], p_test[:,1], 'o', c='r', alpha=.1,label='Test')
    ax[1].xlabel(r'Target $\displaystyle C_{1212}$')
    ax[1].ylabel(r'Prediction $\displaystyle C_{1212}$')
    ax[1].legend()
    ax[1].xlim(left=0)
    ax[1].ylim(bottom=0)
    plt.show()


def parity_main(p_train, t_train, p_test=None, t_test=None):
    #prediction = un_normalize(prediction, truth)

    plt.plot([0, np.max(t_train)], [0, np.max(t_train)], 'k-', lw=2)
    plt.plot(t_train, p_train, 'o', c='b', label='Train')
    #plt.plot(t_test, p_test, 'o', c='r', label='Test')
    plt.legend()
    plt.xlabel(r'Target $\displaystyle C_{1212}$')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.ylabel(r'Prediction $\displaystyle C_{1212}$')
    plt.show()



def prop_clos(base, base1=0, interp=0, extrap=0):
    fig, ax = plt.subplots(2,1)

    ax[0].scatter(base[:,0],base[:,1], s=.5)
    ax[0].set_xlabel(r'$\displaystyle C_{1111}$')
    ax[0].set_ylabel(r'$\displaystyle C_{1212}$')
    ax[0].set_xlim(left=0)
    ax[0].set_ylim(bottom=0)

    ax[1].scatter(base[:,0],base[:,1], s=.5, label='Train Points')
    ax[1].scatter(base1[:,0],base1[:,1],s=.5, c='r', label='Protocol Points')
    ax[1].scatter(interp[:,0],interp[:,1],s=.5, c='r')
    #plt.scatter(extrap[:,0],extrap[:,1],s=25, c='g', label='Extrapolated')
    ax[1].set_xlabel(r'$\displaystyle C_{1111}$')
    ax[1].set_ylabel(r'$\displaystyle C_{1212}$')
    ax[1].set_xlim(left=0)
    ax[1].set_ylim(bottom=0)
    ax[1].legend(markerscale=10)
    plt.tight_layout()
    plt.show()
