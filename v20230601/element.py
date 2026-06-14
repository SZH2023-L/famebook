# element.py
# 功能：计算单元刚度矩阵 Ke 和 对号矩阵 LM，同时返回单元长度和方向余弦
import numpy as np

def compute_Ke_and_LM(nodei, nodej, x, y, Ee, Ae, ndof, nsd):
    """
    输入：
        nodei, nodej: 单元两端节点的全局编号（0-based）
        x, y: 节点坐标数组
        Ee, Ae: 单元弹性模量和截面积
        ndof: 每个节点的自由度数
        nsd: 空间维度 (1或2)
    返回：
        Ke: 单元刚度矩阵 (2×2 或 4×4)
        LM: 单元对号矩阵 (自由度全局编号列表)
        L: 单元长度
        c: x方向余弦
        s: y方向余弦
    """
    if nsd == 1:
        xi, xj = x[nodei], x[nodej]
        L = xj - xi
        c, s = 1.0, 0.0
        k = Ee * Ae / L
        Ke = k * np.array([[1, -1], [-1, 1]])
        LM = np.array([nodei * ndof, nodej * ndof])
    else:  # nsd == 2
        xi, yi = x[nodei], y[nodei]
        xj, yj = x[nodej], y[nodej]
        dx = xj - xi
        dy = yj - yi
        L = np.sqrt(dx**2 + dy**2)
        c = dx / L
        s = dy / L
        k = Ee * Ae / L
        cc, ss, cs = c*c, s*s, c*s
        Ke = k * np.array([
            [cc, cs, -cc, -cs],
            [cs, ss, -cs, -ss],
            [-cc, -cs, cc, cs],
            [-cs, -ss, cs, ss]
        ])
        LM = np.array([
            nodei*ndof + 0, nodei*ndof + 1,
            nodej*ndof + 0, nodej*ndof + 1
        ])
    return Ke, LM, L, c, s