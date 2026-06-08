# 导入numpy库
import numpy as np

# 定义函数：判断矩阵是否奇异（行列式≈0为奇异）
def is_singular(mat, tol=1e-10):
    # 计算矩阵行列式
    det = np.linalg.det(mat)
    # 返回是否奇异
    return abs(det) < tol

# 定义函数：求解桁架结构位移、约束反力、奇异度检查
def solve_truss(K_global, F_global, fixed_dof, fixed_value):
    # 总自由度
    ndof_total = K_global.shape[0]
    # 所有自由度编号
    all_dof = np.arange(ndof_total)
    # 自由自由度 = 总自由度 - 约束自由度
    free_dof = np.setdiff1d(all_dof, fixed_dof)

    # 施加边界前，整体刚度是否奇异
    singular_before = is_singular(K_global)

    # 缩减刚度矩阵：自由-自由
    Kff = K_global[np.ix_(free_dof, free_dof)]
    # 缩减刚度矩阵：自由-约束
    Kfe = K_global[np.ix_(free_dof, fixed_dof)]
    # 已知约束位移
    d_e = fixed_value
    # 自由自由度载荷
    F_f = F_global[free_dof]

    # 求解自由自由度位移
    d_f = np.linalg.solve(Kff, F_f - Kfe @ d_e)

    # 初始化整体位移向量
    d = np.zeros(ndof_total)
    # 赋值自由位移
    d[free_dof] = d_f
    # 赋值约束位移
    d[fixed_dof] = d_e

    # 计算约束反力
    R = K_global @ d - F_global

    # 施加边界后，缩减矩阵是否奇异
    singular_after = is_singular(Kff)

    # 返回位移、反力、奇异度、缩减刚度矩阵Kff
    return d, R, singular_before, singular_after, Kff