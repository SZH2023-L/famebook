# solver.py
# 功能：施加位移边界条件（缩减法），调用LDL^T求解器或直接求解
import numpy as np
from ldlt_solver import ldlt_factor, ldlt_solve, residual_norm

def is_singular(mat, tol=1e-10):
    return abs(np.linalg.det(mat)) < tol

def solve_truss(K_global, F_global, fixed_dof, fixed_value, method="ldlt"):
    """
    求解有限元平衡方程
    返回: d, R, singular_before, singular_after, Kff, L, D, res_norm
    """
    ndof_total = K_global.shape[0]
    all_dof = np.arange(ndof_total)
    free_dof = np.setdiff1d(all_dof, fixed_dof)

    singular_before = is_singular(K_global)

    Kff = K_global[np.ix_(free_dof, free_dof)]
    Kfe = K_global[np.ix_(free_dof, fixed_dof)]
    d_e = fixed_value
    F_f = F_global[free_dof]
    rhs = F_f - Kfe @ d_e

    if method == "reduction":
        d_f = np.linalg.solve(Kff, rhs)
        L, D = None, None
        res_norm = 0.0
    elif method == "ldlt":
        L, D = ldlt_factor(Kff)
        d_f = ldlt_solve(L, D, rhs)
        _, res_norm = residual_norm(Kff, d_f, rhs)
    else:
        raise ValueError(f"未知求解方法: {method}")

    d = np.zeros(ndof_total)
    d[free_dof] = d_f
    d[fixed_dof] = d_e

    R = K_global @ d - F_global
    singular_after = is_singular(Kff)

    return d, R, singular_before, singular_after, Kff, L, D, res_norm