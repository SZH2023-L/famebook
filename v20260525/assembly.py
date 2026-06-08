# 导入numpy库
import numpy as np
# 从element.py中导入单元计算函数，解决之前NameError报错
from element import compute_Ke_and_LM

# 定义函数：组装整体刚度矩阵 + 存储所有单元LM矩阵
# 输入：ndof_total-总自由度，nel-单元总数，IEN-单元节点连接表，x/y-节点坐标，E/A-单元材料属性，ndof-单节点自由度，nsd-维度
def assemble_global_K(ndof_total, nel, IEN, x, y, E, A, ndof, nsd):
    # 初始化整体刚度矩阵，零矩阵，大小为总自由度×总自由度
    K = np.zeros((ndof_total, ndof_total), dtype=float)
    # 初始化空列表，存储所有单元的LM矩阵，满足作业自动生成LM矩阵要求
    LM_all = []
    # 遍历每一个单元，循环次数为单元总数
    for e in range(nel):
        # 读取当前单元的两个节点编号
        ni, nj = IEN[e]
        # 调用函数计算当前单元的Ke、LM、长度、方向余弦
        Ke, LM, _, _, _ = compute_Ke_and_LM(ni, nj, x, y, E[e], A[e], ndof, nsd)
        # 将当前单元的LM矩阵存入总列表
        LM_all.append(LM)
        # 双重循环：对号入座，将单元刚度矩阵叠加到整体刚度矩阵
        for a in range(len(LM)):
            for b in range(len(LM)):
                # 提取整体自由度编号
                I = LM[a]
                J = LM[b]
                # 单元刚度矩阵叠加到整体刚度矩阵对应位置
                K[I, J] += Ke[a, b]
    # 返回整体刚度矩阵、所有单元的LM矩阵
    return K, LM_all