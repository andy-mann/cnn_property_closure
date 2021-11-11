import numpy as np
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def detachData(data):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    assert not torch.is_tensor(data)
    return data

def mae(predictions, truth):
    N = truth.shape[0]
    return 1/N * np.sum(abs(truth - predictions))

def mase(predictions, truth):
    mean = np.mean(truth)
    return mae(predictions, truth) / mean


