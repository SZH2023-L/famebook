# 导入json库，用于读取json输入文件
import json
# 导入numpy库
import numpy as np

# 定义函数：读取json输入文件，返回原始数据字典
# 输入：json_file-json文件路径/名称
def read_json(json_file):
    # 以只读、utf-8编码方式打开json文件
    with open(json_file, 'r', encoding='utf-8') as f:
        # 读取文件并转为python字典
        data = json.load(f)
    # 返回读取到的模型数据
    return data

# 定义函数：模型前处理，解析json数据、转换格式、处理自由度索引
# 输入：data-读取到的json原始数据字典
def preprocess(data):
    # 读取空间维度（1=一维，2=二维）
    nsd = data["nsd"]
    # 读取单个节点的自由度数量
    ndof = data["ndof"]
    # 读取节点总数
    nnp = data["nnp"]
    # 读取单元总数
    nel = data["nel"]
    # 读取单元-节点连接表，并转为numpy数组，同时1基→0基（程序从0开始计数）
    IEN = np.array(data["IEN"]) - 1
    # 读取所有单元的弹性模量，转为numpy数组
    E = np.array(data["E"])
    # 读取所有单元的横截面积，转为numpy数组
    A = np.array(data["CArea"])
    # 读取所有节点的x坐标，转为numpy数组
    x = np.array(data["x"])
    # 如果是二维问题，读取节点y坐标；一维则y=None
    y = np.array(data["y"]) if nsd == 2 else None

    # 读取约束自由度编号，1基→0基
    fixed_dof = np.array(data["fixed_dof"]) - 1
    # 读取约束自由度对应的位移值
    fixed_value = np.array(data["fixed_value"])
    # 读取施加载荷的自由度编号，1基→0基
    force_dof = np.array(data["force_dof"]) - 1
    # 读取载荷对应的大小
    force_value = np.array(data["force_value"])

    # 返回所有前处理后的模型参数
    return nnp, nel, ndof, nsd, x, y, IEN, E, A, fixed_dof, fixed_value, force_dof, force_value