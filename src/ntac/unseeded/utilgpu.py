import math
import numpy as np
from numba import cuda, njit, prange
from scipy.sparse import csr_matrix
from .util import fp_type
import subprocess

import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

def is_cuda_available():
    """Check if CUDA is available on the system"""
    return cuda.is_available() & validate_cuda_toolchain()

def validate_cuda_toolchain():
    from numba import cuda
    import numpy as np
    @cuda.jit
    def dummy_kernel(x):
        idx = cuda.grid(1)
        if idx < x.size:
            x[idx] += 1
    try:
        x = np.zeros(4, dtype=np.float32)
        d_x = cuda.to_device(x)
        dummy_kernel[1, 4](d_x)
        d_x.copy_to_host()
        return True
    except cuda.cudadrv.driver.LinkerError as e:
        print("CUDA driver is available, but PTX linking failed. ")
        print("Make sure you've installed a compatible version of `cudatoolkit`.")
        print("Using conda, you can install it with:")
        print("install -c conda-forge cudatoolkit")
        return False
    except cuda.cudadrv.error.CudaSupportError as e:
        raise RuntimeError("CUDA support error — possibly missing cudatoolkit or incompatible driver.") from e
    except Exception as e:
        raise RuntimeError("Unexpected error during CUDA test kernel execution.") from e

# 1) The CUDA kernel: one thread per (u,i) pair
@cuda.jit
def _wj_cuda(A, M, D, n, m, k):
    u, i = cuda.grid(2)
    if u < n and i < k:
        num = 0.0
        den = 0.0
        # walk the m‐length vectors
        for j in range(m):
            aj = A[u, j]
            bj = M[i, j]
            if aj >= bj:
                den += aj
                num += aj - bj
            else:
                den += bj
                num += bj - aj
        # write result
        D[u, i] = num/den if den != 0.0 else 1.0

def weighted_jaccard_gpu(A: np.ndarray, M: np.ndarray):
    """
    A: (n,m) np.float64 or float64
    M: (k,m) same dtype
    returns D: (n,k) on host
    """
    # shapes
    n, m = A.shape
    k, _ = M.shape

    # 2) copy data to device
    A_dev = cuda.to_device(A.astype(fp_type))
    M_dev = cuda.to_device(M.astype(fp_type))
    D_dev = cuda.device_array((n, k), dtype=A.dtype)

    # 3) choose a reasonable block/grid
    threadsperblock = (16, 16)
    blockspergrid = (
        math.ceil(n / threadsperblock[0]),
        math.ceil(k / threadsperblock[1])
    )

    # 4) launch
    _wj_cuda[blockspergrid, threadsperblock](A_dev, M_dev, D_dev, n, m, k)

    # 5) copy result back
    return D_dev.copy_to_host()

# CPU implementation with Numba optimization
@njit(parallel=True)
def weighted_jaccard_cpu_slow(A, M):
    n, m = A.shape
    k, _ = M.shape
    D = np.zeros((n, k), dtype=A.dtype)
    
    # Use prange for parallel execution of the outer loop
    for u in prange(n):
        for i in range(k):
            num = 0.0
            den = 0.0
            # walk the m‐length vectors
            for j in range(m):
                aj = A[u, j]
                bj = M[i, j]
                if aj >= bj:
                    den += aj
                    num += aj - bj
                else:
                    den += bj
                    num += bj - aj
            # write result
            D[u, i] = num/den if den != 0.0 else 1.0
    
    return D

@njit(parallel=True)
def weighted_jaccard_cpu_fast_aux(indptr, indices, data, sum_A, M, sum_M):
    n = indptr.shape[0] - 1
    k = M.shape[0]
    D = np.empty((n, k), dtype=M.dtype)
    for u in prange(n):
        start, end = indptr[u], indptr[u+1]
        su = sum_A[u]   # now a true scalar
        for i in range(k):
            smax = 0.0
            row_i = M[i]        # speed hack: pull this row outside the inner loop?
            for p in range(start, end):
                j  = indices[p]
                aj = data[p]
                bj = row_i[j]
                if aj > bj:
                    smax += aj - bj
            den = sum_M[i] + smax
            num = sum_M[i] - su + 2*smax
            D[u, i] = num/den if den != 0.0 else 1.0
    return D

def weighted_jaccard_cpu_fast(A, M):
    # usage:
    # A is csr_matrix, M is np.ndarray (n×m dense)
    A = csr_matrix(A)
    sum_A = np.array(A.sum(axis=1)).ravel()   # shape (n,)
    sum_M = M.sum(axis=1)                     # shape (k,)
    return weighted_jaccard_cpu_fast_aux(A.indptr, A.indices, A.data, sum_A, M, sum_M)

@cuda.jit
def _wj_cuda_csr(indptr, indices, data, M, sum_M, sum_A, D, n, k):
    # one thread per (u,i)
    u, i = cuda.grid(2)
    if u >= n or i >= k:
        return

    start = indptr[u]
    end   = indptr[u+1]
    su    = sum_A[u]       # precomputed sum of row u of A
    smax  = 0.0

    # only walk the nnz entries of row u
    for p in range(start, end):
        j  = indices[p]
        aj = data[p]
        bj = M[i, j]        # M is still dense (k×m)
        if aj > bj:
            smax += aj - bj

    # same algebraic rewrite as in the CPU version
    num = sum_M[i] - su + 2*smax
    den = sum_M[i] + smax
    D[u, i] = num/den if den != 0.0 else 1.0


def weighted_jaccard_gpu_csr(A, M):
    A_csr = csr_matrix(A)

    # A_csr: scipy.sparse.csr_matrix on host
    # M: dense np.ndarray, shape (k,m)
    n, m = A_csr.shape
    k, _ = M.shape

    # precompute per-row and per-centroid sums on host
    sum_A = A_csr.sum(axis=1).A1.astype(fp_type)  # shape (n,)
    sum_M = M.sum(axis=1).astype(fp_type)         # shape (k,)

    # copy CSR arrays and M & sums to device
    indptr_dev = cuda.to_device(A_csr.indptr.astype(np.int32))
    indices_dev= cuda.to_device(A_csr.indices.astype(np.int32))
    data_dev   = cuda.to_device(A_csr.data.astype(fp_type))
    M_dev      = cuda.to_device(M.astype(fp_type))
    sum_A_dev  = cuda.to_device(sum_A)
    sum_M_dev  = cuda.to_device(sum_M)
    D_dev      = cuda.device_array((n, k), dtype=fp_type)

    # launch exactly as before
    threadsperblock = (16, 16)
    blockspergrid = (
        math.ceil(n / threadsperblock[0]),
        math.ceil(k / threadsperblock[1])
    )
    _wj_cuda_csr[blockspergrid, threadsperblock](
        indptr_dev, indices_dev, data_dev,
        M_dev, sum_M_dev, sum_A_dev,
        D_dev, n, k
    )

    return D_dev.copy_to_host()
