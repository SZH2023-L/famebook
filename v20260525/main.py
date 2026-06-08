# 导入numpy库
import numpy as np
# 导入模型读取与前处理
from model import read_json, preprocess
# 导入整体刚度组装
from assembly import assemble_global_K
# 导入求解器
from solver import solve_truss
# 导入后处理
from postprocess import post_process

# 运行单个算例并输出完整结果
def run_case(json_name, case_name):
    # 打印算例标题
    print(f"\n==================== {case_name} 结果 ====================")
    # 读取JSON
    data = read_json(json_name)
    # 前处理
    nnp, nel, ndof, nsd, x, y, IEN, E, A, fixed_dof, fixed_value, force_dof, force_value = preprocess(data)
    # 总自由度
    ndof_total = nnp * ndof

    # 组装整体刚度
    K_global, LM_all = assemble_global_K(ndof_total, nel, IEN, x, y, E, A, ndof, nsd)
    print("1. 总体刚度矩阵 K：")
    print(np.round(K_global, 4))
    print(f"刚度矩阵对称性：{np.allclose(K_global, K_global.T)}")

    # 组装载荷向量
    F_global = np.zeros(ndof_total)
    for dof, val in zip(force_dof, force_value):
        F_global[dof] = val

    # 求解（现在多返回 Kff）
    d, R, singular_before, singular_after, Kff = solve_truss(K_global, F_global, fixed_dof, fixed_value)
    print(f"\n2. 节点位移 d：\n{np.round(d, 6)}")

    # 输出算例2节点3位移u3、v3
    if nsd == 2:
        u3 = d[4]
        v3 = d[5]
        print(f"\n 节点3位移：u3 = {u3:.6f}，v3 = {v3:.6f}")

    # 输出反力
    print(f"\n3. 约束反力 R：\n{np.round(R, 6)}")
    # 奇异度检查
    print(f"\n4. 奇异度检查：")
    print(f"   施加边界前总体刚度矩阵奇异：{singular_before}")
    print(f"   施加边界后缩减刚度矩阵奇异：{singular_after}")

    # ====================== 新增：打印缩减刚度矩阵 Kff ======================
    print(f"   施加边界后缩减刚度矩阵 Kff = ")
    print(np.round(Kff, 4))
    # ======================================================================

    # 后处理
    stress, force, L, c, s = post_process(d, IEN, LM_all, x, y, E, A, nsd)
    print(f"\n5. 单元后处理：")
    for i in range(nel):
        print(f"   单元{i+1}：长度={L[i]:.4f}, c={c[i]:.4f}, s={s[i]:.4f}, 应力={stress[i]:.6f}, 轴力={force[i]:.4f}")

    # 二维算例输出LM矩阵
    if nsd == 2:
        print(f"\n自动生成LM矩阵：")
        for idx, lm in enumerate(LM_all):
            print(f"单元{idx+1} LM = {lm}")
    return

# 主程序入口
if __name__ == "__main__":
    # 运行算例1
    run_case("input_1d_bar.json", "算例1：一维两杆桁架")
    # 运行算例2
    run_case("input_2d_truss.json", "算例2：二维两杆桁架")