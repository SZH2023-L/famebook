# ldlt_solver.py
# 任务1：实现对称正定矩阵的 LDL^T 分解和求解
import numpy as np

def ldlt_factor(K):
    """LDL^T分解，返回 L (单位下三角) 和 D (对角数组)"""
    K = np.array(K, dtype=float)
    n = K.shape[0]
    L = np.eye(n)
    D = np.zeros(n)
    A = K.copy()
    for j in range(n):
        d = A[j, j]
        if d <= 1e-12:
            raise ValueError(f"零主元或非正主元，位置 j={j}, D={d}")
        D[j] = d
        for i in range(j+1, n):
            L[i, j] = A[i, j] / d
        for i in range(j+1, n):
            for k in range(j+1, n):
                A[i, k] -= L[i, j] * D[j] * L[k, j]
    return L, D

def ldlt_solve(L, D, R):
    """解 LDL^T x = R"""
    n = len(D)
    # 前代：L y = R
    y = np.zeros(n)
    for i in range(n):
        s = R[i]
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s
    # 对角：D z = y
    z = y / D
    # 回代：L^T x = z
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        s = z[i]
        for j in range(i+1, n):
            s -= L[j, i] * x[j]
        x[i] = s
    return x

def residual_norm(K, a, R):
    """计算残差 r = R - K a 及其二范数"""
    r = R - K @ a
    return r, np.linalg.norm(r)

def condition_number(K):
    """条件数估计 (适用于小规模矩阵)"""
    eigvals = np.linalg.eigvalsh(K)
    return eigvals[-1] / eigvals[0]