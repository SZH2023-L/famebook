# external_solver.py
# 任务3：封装外部稀疏求解器（优先 Intel MKL PARDISO，降级 SciPy SuperLU）
import numpy as np
import time
import scipy.sparse as sp

def solve_sparse(K, R, solver_name="pardiso"):
    """
    求解稀疏线性方程组 K x = R
    返回: x, 求解时间, 实际使用的求解器名称
    """
    if solver_name == "pardiso":
        try:
            from pypardiso import spsolve
            print("   [求解器] 使用 PyPARDISO (Intel MKL PARDISO)")
            start = time.perf_counter()
            x = spsolve(K, R)
            elapsed = time.perf_counter() - start
            return x, elapsed, "PyPARDISO (MKL PARDISO)"
        except ImportError:
            print("   [警告] 未安装 pypardiso，降级使用 scipy.sparse.linalg.spsolve")
            solver_name = "scipy"

    if solver_name == "scipy":
        from scipy.sparse.linalg import spsolve
        print("   [求解器] 使用 scipy.sparse.linalg.spsolve (SuperLU)")
        if not isinstance(K, sp.csc_matrix):
            K = K.tocsc()
        start = time.perf_counter()
        x = spsolve(K, R)
        elapsed = time.perf_counter() - start
        return x, elapsed, "SciPy SuperLU"

    raise ValueError(f"不支持的求解器: {solver_name}")