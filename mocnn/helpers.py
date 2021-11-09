import numpy as np
import torch


def un_normalize(predictions, truth):
    unnorm = (predictions * (np.max(truth) - np.min(truth))) + np.min(truth)
    return unnorm


def dataset_to_np(dataset):
    # makes unzipping easier somehow?
    X, y = zip(*[(detachData(di[0]),
                  detachData(di[1])) for i, di in enumerate(dataset)])
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

def mase(predictions, truth):
    N = truth.shape[0]
    mean = np.mean(predictions)
    return 1/N * np.sum(abs(predictions-truth) / mean)


def mae(predictions, truth):
    N = truth.shape[0]
    return 1/N * np.sum(abs(predictions - truth))