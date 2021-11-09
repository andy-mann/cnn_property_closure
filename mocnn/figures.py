import matplotlib.pyplot as plt
import numpy as np
import os
from mocnn.helpers import *


def pred_vs_truth(prediction, truth, fpath):
    prediction = un_normalize(prediction, truth)

    plt.scatter(truth, prediction)
    plt.xlabel('True Stifness')
    plt.ylabel('Predicted Stiffness')
    plt.savefig(f'{fpath}/output')

