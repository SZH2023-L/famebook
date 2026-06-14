计算力学2.4作业运行说明

作业简介
本程序是计算力学课程"有限元平衡方程组的求解与误差分析"作业的完整实现，包含对称正定矩阵LDLᵀ分解求解器、误差分析、稀疏求解器调用及多个验证算例，与2.3作业有限元框架无缝衔接。

环境要求
Python 3.8+
 依赖包：外部调用求解器安装`pypardiso`，

文件结构
```
.
├── main.py                 # 主程序，统一调度所有算例
├── model.py                # 前处理：读取JSON输入，索引转换
├── element.py              # 单元分析：计算单元刚度矩阵和LM矩阵
├── assembly.py             # 总体组装：组装全局刚度矩阵
├── ldlt_solver.py          # 核心：自行实现LDLᵀ分解与求解
├── solver.py               # 边界条件处理与求解器调度
├── external_solver.py      # 外部稀疏求解器封装(PyPARDISO/SuperLU)
├── postprocess.py          # 后处理：计算单元应力、轴力
├── poisson_fem.py          # 二维Poisson方程有限元求解(T3单元)
├── test_tridiagonal.py     # 三对角矩阵算例
├── test_nonpositive.py     # 非正定矩阵检测
├── input_1d_bar.json       # 算例0-1：一维两杆输入
└── input_2d_truss.json     # 算例0-2：二维两杆输入
```

运行方法
1. 准备工作
   确保所有文件在同一目录下
   确认文件名与上述结构一致
   安装所有依赖包

2.模块复用说明：
model.py：读取 JSON 输入文件，进行前处理（节点、单元、材料、边界条件等）。
element.py：计算单元刚度矩阵 Ke 和对号矩阵 LM。
assembly.py：组装总体刚度矩阵 K 和存储所有单元的 LM。
postprocess.py：后处理计算单元应力、轴力等。

3.求解器运行环境说明
CPU：Intel Core i9-13900H 2.60GHz
内存:8 GB
OS：Windows 11
Python 3.12.7（Anaconda）
求解器版本：pypardiso 0.4.7（基于 Intel MKL 2024）

4. 运行全部算例
   ```bash
   python main.py
   ```
   程序将依次运行：
    算例0：一维两杆、二维两杆桁架
   算例1：三对角对称正定矩阵
   任务2：病态矩阵误差分析
   算例2：非正定矩阵检测
   算例4：二维Poisson方程(50×50/100×100/200×200/500×500/1000×1000网格)

输出说明
控制台输出：所有算例的详细结果，包括矩阵信息、求解时间、位移、应力、残差、误差等
图片文件：Poisson方程算例会生成`poisson_nx×ny.png`，包含数值解、精确解和误差云图

