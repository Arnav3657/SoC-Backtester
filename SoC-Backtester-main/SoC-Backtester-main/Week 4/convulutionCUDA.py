@cuda.jit
def convolution_cuda(a, b, result):
    i = cuda.grid(1)
    m = b.size
    if i < result.size:
        tmp = 0.0
        for j in range(m):
            tmp += a[i + j] * b[j]
        result[i] = tmp

a = np.random.rand(100000).astype(np.float32)
b = np.random.rand(100).astype(np.float32)
result = np.zeros(a.size - b.size + 1, dtype=np.float32)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_result = cuda.device_array(result.size, dtype=np.float32
threads_per_block = 256
blocks_per_grid = math.ceil(result.size / threads_per_block)

def run_convolution_cuda():
    convolution_cuda[blocks_per_grid, threads_per_block](d_a d_b, d_result)
    cuda.synchronize()

%timeit run_convolution_cuda()
