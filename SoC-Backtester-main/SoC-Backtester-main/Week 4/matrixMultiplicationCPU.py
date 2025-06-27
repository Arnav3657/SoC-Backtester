import numpy as np


N = 512
A = np.random.rand(N, N)
B = np.random.rand(N, N)

# direct matrix multiplication function
def matmul_cpu(A, B):
    return np.dot(A, B)


C = matmul_cpu(A, B)
