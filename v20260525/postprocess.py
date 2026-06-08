# 导入numpy库
import numpy as np
# 从element.py导入单元计算函数
from element import compute_Ke_and_LM

# 定义函数：后处理，计算每个单元的应力、轴力、长度、方向余弦
# 输入：d-整体位移向量，IEN-单元连接表，LM_all-所有单元LM矩阵，x/y-节点坐标，E/A-材料属性，nsd-维度
def post_process(d, IEN, LM_all, x, y, E, A, nsd):
    # 获取单元总数
    nel = len(IEN)
    # 初始化空列表，存储每个单元的应力
    stress_list = []
    # 初始化空列表，存储每个单元的轴力
    force_list = []
    # 初始化空列表，存储每个单元的长度
    length_list = []
    # 初始化空列表，存储每个单元x方向方向余弦
    cos_list = []
    # 初始化空列表，存储每个单元y方向方向余弦
    sin_list = []

    # 遍历每一个单元，计算后处理数据
    for e in range(nel):
        # 读取当前单元的两个节点
        ni, nj = IEN[e]
        # 读取当前单元的LM矩阵
        LM = LM_all[e]
        # 提取当前单元的节点位移
        de = d[LM]
        # 调用函数获取单元几何参数
        Ke, _, L, c, s = compute_Ke_and_LM(ni, nj, x, y, E[e], A[e], 1, nsd)

        # 一维单元应力计算公式
        if nsd == 1:
            sigma = E[e]/L * (de[1] - de[0])
        # 二维单元应力计算公式【重点修改：修正符号，匹配作业要求】
        else:
            sigma = E[e]/L * (c*de[0] + s*de[1] - c*de[2] - s*de[3])
        # 计算单元轴力 = 应力 × 横截面积
        N = sigma * A[e]

        # 将当前单元应力存入列表
        stress_list.append(sigma)
        # 将当前单元轴力存入列表
        force_list.append(N)
        # 将当前单元长度存入列表
        length_list.append(L)
        # 将当前单元x方向余弦存入列表
        cos_list.append(c)
        # 将当前单元y方向余弦存入列表
        sin_list.append(s)
    # 返回所有单元的后处理数据
    return stress_list, force_list, length_list, cos_list, sin_list