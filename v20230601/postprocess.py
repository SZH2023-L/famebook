# postprocess.py
# 功能：后处理，计算每个单元的应力、轴力、长度、方向余弦
# 导入numpy库用于数值计算
import numpy as np
# 从element模块导入单元几何和刚度计算函数（复用其中的长度、方向余弦）
from element import compute_Ke_and_LM

def post_process(d, IEN, LM_all, x, y, E, A, nsd):
    """
    后处理主函数
    输入：
        d: 全局位移向量（一维数组，长度为总自由度数）
        IEN: 单元-节点连接表（nel × 2，每个单元两个节点编号，0-based）
        LM_all: 每个单元的LM矩阵列表（每个LM为单元自由度全局编号的数组）
        x, y: 节点坐标数组（一维为x，二维为x和y）
        E, A: 单元弹性模量和截面积数组
        nsd: 空间维度（1或2）
    返回：
        stress_list: 每个单元的应力值列表
        force_list: 每个单元的轴力值列表
        length_list: 每个单元的长度列表
        cos_list: 每个单元的x方向余弦列表
        sin_list: 每个单元的y方向余弦列表
    """
    # 获取单元总数
    nel = len(IEN)
    # 初始化存储列表
    stress_list, force_list, length_list, cos_list, sin_list = [], [], [], [], []
    # 遍历每个单元
    for e in range(nel):
        # 获取当前单元的两个节点编号（0-based）
        ni, nj = IEN[e]
        # 获取当前单元的LM矩阵（单元自由度全局编号）
        LM = LM_all[e]
        # 提取当前单元节点的位移（按LM顺序排列）
        de = d[LM]
        # 调用element模块的函数，获得长度L、方向余弦c、s（Ke和LM忽略）
        _, _, L, c, s = compute_Ke_and_LM(ni, nj, x, y, E[e], A[e], 1, nsd)
        # 根据空间维度计算应力
        if nsd == 1:
            # 一维杆单元：应力 = E/L * (uj - ui)
            sigma = E[e] / L * (de[1] - de[0])
        else:
            # 二维桁架单元：标准物理公式
            # 应力 = E/L * (c * (uj - ui) + s * (vj - vi))
            # 其中 de = [ui, vi, uj, vj]
            sigma = E[e] / L * (c*de[0] + s*de[1] - c*de[2] - s*de[3])
        # 计算轴力 = 应力 × 截面积
        N = sigma * A[e]
        # 将结果存入列表
        stress_list.append(sigma)
        force_list.append(N)
        length_list.append(L)
        cos_list.append(c)
        sin_list.append(s)
    # 返回所有单元的后处理数据
    return stress_list, force_list, length_list, cos_list, sin_list