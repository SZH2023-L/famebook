# main.py

MESH_SIZES = [50, 100, 200, 500, 1000]   # 算例4的网格规模，可修改

import numpy as np
import time
from model import read_json, preprocess
from assembly import assemble_global_K
from solver import solve_truss
from postprocess import post_process
from poisson_fem import solve_poisson
from ldlt_solver import ldlt_factor, ldlt_solve, condition_number

np.set_printoptions(linewidth=200, threshold=np.inf, precision=4, suppress=True)

# ============================================================================
# 算例0：两个桁架结构
# ============================================================================
def run_truss_case(json_name, case_name, solver_method="ldlt"):
    print(f"\n{'='*50}\n{case_name}\n{'='*50}")
    data = read_json(json_name)
    nnp, nel, ndof, nsd, x, y, IEN, E, A, fixed_dof, fixed_value, force_dof, force_value = preprocess(data)
    ndof_total = nnp * ndof
    print(f"矩阵阶数 n = {ndof_total}")

    K_global, LM_all = assemble_global_K(ndof_total, nel, IEN, x, y, E, A, ndof, nsd)

    print("\n1. 总体刚度矩阵 K:")
    print(np.round(K_global, 4))
    print(f"\n刚度矩阵对称性: {np.allclose(K_global, K_global.T)}")

    F_global = np.zeros(ndof_total)
    for dof, val in zip(force_dof, force_value):
        F_global[dof] = val

    start_time = time.perf_counter()
    d, R, sing_before, sing_after, Kff, L, D, res_norm = solve_truss(
        K_global, F_global, fixed_dof, fixed_value, method=solver_method)
    elapsed = time.perf_counter() - start_time
    print(f"\n求解时间: {elapsed:.6f} 秒")

    if solver_method == "ldlt" and L is not None:
        print("LDL^T 分解成功完成")
        min_pivot = np.min(D)
        print(f"主元最小值: {min_pivot:.4e}")

    print("\n2. 节点位移 d (解向量):")
    print(np.round(d, 6))

    if nsd == 2 and nnp == 3:
        u3 = d[4]
        v3 = d[5]
        print(f"\n节点3位移：u3 = {u3:.6f}, v3 = {v3:.6f}")

    print("\n3. 约束反力 R:")
    print(np.round(R, 6))

    print("\n4. 奇异度检查:")
    print(f"施加边界前总体刚度矩阵奇异: {sing_before}")
    print(f"施加边界后缩减刚度矩阵奇异: {sing_after}")
    print("施加边界后缩减刚度矩阵 Kff = ")
    print(np.round(Kff, 4))

    print(f"\n缩减方程残差范数 ||RHS - Kff*d_f|| = {res_norm:.4e}")

    stress, force, L_elem, c, s = post_process(d, IEN, LM_all, x, y, E, A, nsd)
    print("\n5. 单元后处理:")
    for i in range(nel):
        print(f"   单元{i+1}: 长度={L_elem[i]:.4f}, c={c[i]:.4f}, s={s[i]:.4f}, 应力={stress[i]:.6f}, 轴力={force[i]:.4f}")

    if nsd == 2:
        print("\n自动生成LM矩阵：")
        for idx, lm in enumerate(LM_all):
            print(f"单元{idx+1} LM = {lm}")

# ============================================================================
# 算例1：三对角对称正定矩阵
# ============================================================================
def generate_tridiagonal(n):
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 2.0
        if i > 0:
            K[i, i-1] = -1.0
            K[i-1, i] = -1.0
    x_exact = np.ones(n)
    R = K @ x_exact
    return K, R, x_exact

def run_tridiagonal():
    print("\n" + "="*70)
    print("算例1：三对角对称正定矩阵 (LDL^T 求解)")
    print("="*70)
    sizes = [10, 100, 500, 1000]
    print(f"{'n':>6} {'时间(秒)':>12} {'相对误差':>14} {'稠密内存(MB)':>16} {'稀疏内存(MB)':>16}")
    print("-" * 70)

    times = []
    for n in sizes:
        print(f"正在计算 n={n} ...", end='', flush=True)
        try:
            K, R, x_exact = generate_tridiagonal(n)
            start = time.perf_counter()
            L, D = ldlt_factor(K)
            x = ldlt_solve(L, D, R)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            rel_err = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
            dense_mem = (n * n * 8) / (1024 ** 2)
            nnz = 3 * n - 2
            sparse_mem = (nnz * 8) / (1024 ** 2)
            print(f"\r{n:6d} {elapsed:12.6f} {rel_err:14.2e} {dense_mem:16.2f} {sparse_mem:16.2f}")
            print(f"   解向量前5个元素: {x[:5]}")
            r = R - K @ x
            r_norm = np.linalg.norm(r)
            print(f"   残差范数 ||r|| = {r_norm:.4e}")
        except Exception as e:
            print(f"\r{n:6d} 失败: {e}")
            times.append(None)

    print("\n趋势分析：")
    valid_times = [t for t in times if t is not None]
    if len(valid_times) >= 2:
        for i in range(1, len(sizes)):
            if times[i] is not None and times[i-1] is not None:
                ratio = times[i] / times[i-1]
                size_ratio = sizes[i] / sizes[i-1]
                print(f"   n从 {sizes[i-1]} 增加到 {sizes[i]}，时间增加 {ratio:.2f} 倍，规模增加 {size_ratio:.0f} 倍")
        print("   由于 LDL^T 分解复杂度约为 O(n^3)，时间增长应接近规模比的立方。")
    print("\n算例1完成。")

# ============================================================================
# 任务2：病态矩阵分析（误差、残差、条件数）
# ============================================================================
def run_ill_conditioned():
    print("\n" + "="*70)
    print("任务2：病态矩阵分析 (条件数、残差、误差)")
    print("="*70)
    
    # 构造病态矩阵和精确解
    K = np.array([[1.0, 1.0],
                  [1.0, 1.0001]], dtype=np.float64)
    a_exact = np.array([1.0, 1.0])
    R = K @ a_exact
    
    # 条件数
    cond = condition_number(K)
    print(f"矩阵 K = \n{K}")
    print(f"条件数 cond(K) = {cond:.4e}")
    
    # 1. 双精度计算
    L, D = ldlt_factor(K)
    a_double = ldlt_solve(L, D, R)
    r_double = R - K @ a_double
    rel_res_double = np.linalg.norm(r_double) / np.linalg.norm(R)
    rel_err_double = np.linalg.norm(a_double - a_exact) / np.linalg.norm(a_exact)
    
    print("\n--- 双精度计算 ---")
    print(f"数值解 a = {a_double}")
    print(f"残差 r = {r_double}")
    print(f"相对残差 = {rel_res_double:.4e}")
    print(f"相对误差 = {rel_err_double:.4e}")
    
    # 2. 人为四舍五入到4位有效数字（模拟低精度）
    K_4digit = np.round(K, 4)
    R_4digit = np.round(R, 4)
    try:
        L4, D4 = ldlt_factor(K_4digit)
        a_4digit = ldlt_solve(L4, D4, R_4digit)
        # 用原始精确的 K 和 R 计算残差和误差（公平比较）
        r_4digit = R - K @ a_4digit
        rel_res_4digit = np.linalg.norm(r_4digit) / np.linalg.norm(R)
        rel_err_4digit = np.linalg.norm(a_4digit - a_exact) / np.linalg.norm(a_exact)
        print("\n--- 四舍五入到4位有效数字 ---")
        print(f"数值解 a = {a_4digit}")
        print(f"残差 r = {r_4digit}")
        print(f"相对残差 = {rel_res_4digit:.4e}")
        print(f"相对误差 = {rel_err_4digit:.4e}")
    except ValueError as e:
        print(f"\n四舍五入矩阵分解失败: {e}")
    
    # 3. 单精度 (float32) 计算（可选）
    K_single = K.astype(np.float32)
    R_single = R.astype(np.float32)
    try:
        Ls, Ds = ldlt_factor(K_single)
        a_single = ldlt_solve(Ls, Ds, R_single)
        r_single = R - K @ a_single   # 用原始双精度K,R计算
        rel_res_single = np.linalg.norm(r_single) / np.linalg.norm(R)
        rel_err_single = np.linalg.norm(a_single - a_exact) / np.linalg.norm(a_exact)
        print("\n--- 单精度 (float32) ---")
        print(f"数值解 a = {a_single}")
        print(f"残差 r = {r_single}")
        print(f"相对残差 = {rel_res_single:.4e}")
        print(f"相对误差 = {rel_err_single:.4e}")
    except Exception as e:
        print(f"\n单精度计算失败: {e}")
    
    print("\n结论：")
    print("病态问题中，条件数很大（此处约4e4），输入数据的微小误差（如四舍五入）")
    print("会导致解的相对误差被放大数千倍。即使残差很小，解也可能极不准确。")
    print("因此，对于病态方程组，不能单凭残差判断解的精度。")
    print("\n任务2完成。")

# ============================================================================
# 算例2：非正定矩阵检测
# ============================================================================
def run_nonpositive():
    print("\n" + "="*70)
    print("算例2：非正定矩阵检测 (LDL^T 分解)")
    print("="*70)
    K = np.array([[1.0, 2.0], [2.0, 1.0]])
    R = np.array([1.0, 1.0])
    print("测试矩阵 K =")
    print(K)
    print(f"右端项 R = {R}")
    eigvals = np.linalg.eigvalsh(K)
    print(f"矩阵特征值: {eigvals}，存在负特征值 → 非正定")
    try:
        L, D = ldlt_factor(K)
        print("\n错误：分解未检测到非正主元，但矩阵应为非正定！")
    except ValueError as e:
        print(f"\n正确捕获异常: {e}")
        print("程序已停止 LDL^T 求解，符合要求。")
    print("\n说明：")
    print("有限元刚度矩阵在未施加足够位移边界条件时出现零主元（奇异），")
    print("是因为整体结构存在刚体位移自由度，导致平衡方程不唯一解。")
    print("此时矩阵半正定，但至少有一个特征值为零，对应刚体运动模式。")
    print("\n算例2完成。")

# ============================================================================
# 主函数
# ============================================================================
def main():
    # 算例0
    run_truss_case("input_1d_bar.json", "算例0-1：一维两杆桁架", solver_method="ldlt")
    run_truss_case("input_2d_truss.json", "算例0-2：二维两杆桁架", solver_method="ldlt")
    
    # 算例1
    run_tridiagonal()
    
    # 任务2（病态矩阵分析）
    run_ill_conditioned()
    
    # 算例2
    run_nonpositive()
    
    # 算例4
    print("\n" + "="*60)
    print(f"算例4: 二维 Poisson 方程 (T3单元) 将运行以下网格规模: {MESH_SIZES}")
    print("="*60)
    for nx in MESH_SIZES:
        solve_poisson(nx, nx, solver_name="pardiso")

if __name__ == "__main__":
    main()