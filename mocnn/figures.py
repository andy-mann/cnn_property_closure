import matplotlib.pyplot as plt
import numpy as np
import os
from mocnn.helpers import *


def parity(prediction, truth, component, model, fpath):
    prediction = un_normalize(prediction, truth)

    plt.scatter(truth, prediction)
    plt.xlabel(f'True {component}')
    plt.ylabel(f'Predicted {component}')
    plt.savefig(f'{fpath}/output/{model}_{component}.png')

