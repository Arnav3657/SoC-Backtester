from numba import cuda
import math

@cuda.jit
def moving_average_cuda(arr, window, result):
    i = cuda.grid(1)
    if i < result.size:
        temp = 0.0
        for j in range(window):
            temp += arr[i + j]
        result[i] = temp / window

# Setup:
x = np.random.rand(100000).astype(np.float32)
w = 100
result = np.zeros(x.size - w + 1, dtype=np.float32)

d_x = cuda.to_device(x)
d_result = cuda.device_array(x.size - w + 1, dtype=np.float32)

threads_per_block = 256
blocks_per_grid = math.ceil(result.size / threads_per_block)

def run_moving_average_cuda():
    moving_average_cuda[blocks_per_grid, threads_per_block](d_x, w, d_result)
    cuda.synchronize()

# %timeit run_moving_average_cuda()
