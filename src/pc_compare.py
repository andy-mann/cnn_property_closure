import numpy as np
import matplotlib.pyplot as plt
import pickle

In [252]: a = np.load("bigger_interp.npy")

In [253]: a.shape
Out[253]: (1, 31, 31, 31)

In [254]: b = np.load("generated_stats.npy")

In [255]: b.shape
Out[255]: (100, 31, 31, 31)

In [256]: a[0, 0, 0, 0]
Out[256]: (0.5493269779463599+0j)

In [257]: a[0, 15,15,15]
Out[257]: (0.7399214527877547+0j)

In [258]: a = np.reshape(a, (1, 31 ** 3))

In [259]: b = b[0][None]

In [260]: b.shape
Out[260]: (1, 31, 31, 31)

In [261]: b = np.reshape(b, (1, 31 ** 3))

In [262]: m_a = a - mean[None]

In [263]: m_b = b - mean[None]