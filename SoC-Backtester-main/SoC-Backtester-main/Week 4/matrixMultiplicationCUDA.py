import numpy as np
from numba import cuda
import math


N = 512
A = np.random.rand(N N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

@cuda.jit
def matmul_kernel(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C = cuda.device_array((N, N), dtype=np.float32)


threads_per_block = (16, 16)
blocks_per_grid_x = math.ceil(A.shape[0] / threads_per_block[0])
blocks_per_grid_y = math.ceil(B.shape[1] / threads_per_block[1])


matmul_kernel[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](d_A, d_B, d_C)
C_result = d_C.copy_to_host()


%timeit np.dot(A, B)


def run_gpu():
    matmul_kernel[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](d_A, d_B, d_C)
    cuda.synchronize()
%timeit run_gpu()
