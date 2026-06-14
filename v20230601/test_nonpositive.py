# test_nonpositive.py
# 算例2：非正定矩阵检测
# 测试矩阵 K = [[1,2],[2,1]]，应为非正定（特征值 -1, 3）
# LDL^T 分解时应检测到非正主元并抛出异常

import numpy as np
from ldlt_solver import ldlt_factor

def test_nonpositive():
    print("\n" + "="*70)
    print("算例2：非正定矩阵检测 (LDL^T 分解)")
    print("="*70)
    
    # 构造非正定矩阵
    K = np.array([[1.0, 2.0],
                  [2.0, 1.0]])
    R = np.array([1.0, 1.0])
    
    print("测试矩阵 K =")
    print(K)
    print(f"右端项 R = {R}")
    
    # 检查特征值（可选）
    eigvals = np.linalg.eigvalsh(K)
    print(f"矩阵特征值: {eigvals}，存在负特征值 → 非正定")
    
    # 尝试 LDL^T 分解，应捕获 ValueError
    try:
        L, D = ldlt_factor(K)
        print("\n错误：分解未检测到非正主元，但矩阵应为非正定！")
    except ValueError as e:
        print(f"\n正确捕获异常: {e}")
        print("程序已停止 LDL^T 求解，符合要求。")
    
    # 补充说明：缺少位移边界条件时刚度矩阵奇异的原因
    print("\n说明：")
    print("有限元刚度矩阵在未施加足够位移边界条件时出现零主元（奇异），")
    print("是因为整体结构存在刚体位移自由度，导致平衡方程不唯一解。")
    print("此时矩阵半正定，但至少有一个特征值为零，对应刚体运动模式。")

if __name__ == "__main__":
    test_nonpositive()