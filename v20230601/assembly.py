# 功能：组装总体刚度矩阵，存储所有单元的LM矩阵
import numpy as np
from element import compute_Ke_and_LM

def assemble_global_K(ndof_total, nel, IEN, x, y, E, A, ndof, nsd):
    """
    组装总体刚度矩阵 K (ndof_total × ndof_total)
    同时返回每个单元的LM矩阵列表
    """
    K = np.zeros((ndof_total, ndof_total), dtype=float)
    LM_all = []   # 存储每个单元的LM
    for e in range(nel):
        ni, nj = IEN[e]
        Ke, LM, _, _, _ = compute_Ke_and_LM(ni, nj, x, y, E[e], A[e], ndof, nsd)
        LM_all.append(LM)
        # 直接组装：根据LM将Ke叠加到K
        for a in range(len(LM)):
            for b in range(len(LM)):
                I = LM[a]
                J = LM[b]
                K[I, J] += Ke[a, b]
    return K, LM_all