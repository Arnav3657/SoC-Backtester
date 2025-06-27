import numpy as np
#built in moving average
def moving_average_cpu(arr, window):
    return np.convolve(arr, np.ones(window), 'valid') / window


x = np.random.rand(100000).astype(np.float32)
w = 100
# %timeit moving_average_cpu(x, w)
