# 导入numpy库，用于数值计算、矩阵运算
import numpy as np

# 定义函数：计算单元刚度矩阵Ke + 单元对号矩阵LM + 单元几何信息(长度、方向余弦)
# 输入参数：nodei-单元i端节点号，nodej-单元j端节点号，x-节点x坐标数组，y-节点y坐标数组
#           Ee-单元弹性模量，Ae-单元横截面积，ndof-单节点自由度，nsd-空间维度(1D/2D)
def compute_Ke_and_LM(nodei, nodej, x, y, Ee, Ae, ndof, nsd):
    # 判断是否为一维问题
    if nsd == 1:
        # 读取i、j节点的x坐标
        xi, xj = x[nodei], x[nodej]
        # 计算一维单元长度
        L = xj - xi
        # 一维单元方向余弦：x方向为1，y方向为0
        c, s = 1.0, 0.0
    # 否则为二维问题
    else:
        # 读取i节点x、y坐标
        xi, yi = x[nodei], y[nodei]
        # 读取j节点x、y坐标
        xj, yj = x[nodej], y[nodej]
        # 计算单元x方向坐标差
        dx = xj - xi
        # 计算单元y方向坐标差
        dy = yj - yi
        # 用勾股定理计算二维单元长度
        L = np.sqrt(dx**2 + dy**2)
        # 计算x方向方向余弦
        c = dx / L
        # 计算y方向方向余弦
        s = dy / L

    # 计算单元刚度系数 k = E*A/L
    k = Ee * Ae / L
    # 一维单元刚度矩阵计算逻辑
    if nsd == 1:
        # 构建一维杆单元刚度矩阵(2×2)
        Ke = k * np.array([[1, -1],
                           [-1, 1]])
        # 构建一维单元对号矩阵LM：i节点自由度、j节点自由度
        LM = np.array([nodei * ndof, nodej * ndof])
    # 二维单元刚度矩阵计算逻辑
    else:
        # 计算方向余弦平方 c²
        cc, ss, cs = c**2, s**2, c*s
        # 构建二维桁架单元刚度矩阵(4×4)
        Ke = k * np.array([
            [cc, cs, -cc, -cs],
            [cs, ss, -cs, -ss],
            [-cc, -cs, cc, cs],
            [-cs, -ss, cs, ss]
        ])
        # 构建二维单元对号矩阵LM：i节点u、v自由度，j节点u、v自由度
        LM = np.array([
            nodei*ndof + 0, nodei*ndof + 1,
            nodej*ndof + 0, nodej*ndof + 1
        ])
    # 返回单元刚度矩阵、对号矩阵、单元长度、x方向余弦、y方向余弦
    return Ke, LM, L, c, s