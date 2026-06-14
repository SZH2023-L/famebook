# model.py
# 功能：读取JSON输入文件，进行前处理，将1-based索引转换为0-based
import json
import numpy as np

def read_json(json_file):
    """读取JSON文件，返回数据字典"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess(data):
    """
    从JSON数据中提取有限元模型信息，并转换索引为0-based
    返回：
        nnp: 节点总数
        nel: 单元总数
        ndof: 每个节点的自由度数
        nsd: 空间维度 (1或2)
        x, y: 节点坐标 (y为一维时返回None)
        IEN: 单元-节点连接表 (0-based)
        E, A: 弹性模量、截面积数组
        fixed_dof, fixed_value: 约束自由度编号和位移值 (0-based)
        force_dof, force_value: 施加载荷的自由度编号和载荷值 (0-based)
    """
    nsd = data["nsd"]
    ndof = data["ndof"]
    nnp = data["nnp"]
    nel = data["nel"]
    # 读取IEN，并减去1转为0-based索引
    IEN = np.array(data["IEN"]) - 1
    E = np.array(data["E"])
    A = np.array(data["CArea"])
    x = np.array(data["x"])
    y = np.array(data["y"]) if nsd == 2 else None

    fixed_dof = np.array(data["fixed_dof"]) - 1
    fixed_value = np.array(data["fixed_value"])
    force_dof = np.array(data["force_dof"]) - 1
    force_value = np.array(data["force_value"])

    return nnp, nel, ndof, nsd, x, y, IEN, E, A, fixed_dof, fixed_value, force_dof, force_value