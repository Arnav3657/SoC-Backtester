def convolution_cpu(a, b):
    return np.convolve(a, b, mode='valid')


a = np.random.rand(100000).astype(np.float32)
b = np.random.rand(100).astype(np.float32)
 %timeit convolution_cpu(a, b)
