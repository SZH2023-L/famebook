import numpy as np
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt
from external_solver import solve_sparse

def u_exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f_source(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def generate_triangle_mesh(nx, ny, Lx=1.0, Ly=1.0):
    npx = nx + 1
    npy = ny + 1
    x_coord = np.linspace(0, Lx, npx)
    y_coord = np.linspace(0, Ly, npy)
    nodes = np.zeros((npx * npy, 2))
    for j in range(npy):
        for i in range(npx):
            nodes[j * npx + i] = [x_coord[i], y_coord[j]]
    nel = 2 * nx * ny
    IEN = np.zeros((nel, 3), dtype=int)
    elem = 0
    for j in range(ny):
        for i in range(nx):
            n0 = j * npx + i
            n1 = j * npx + i + 1
            n2 = (j+1) * npx + i
            n3 = (j+1) * npx + i + 1
            IEN[elem] = [n0, n1, n2]
            elem += 1
            IEN[elem] = [n1, n3, n2]
            elem += 1
    return nodes, IEN

def assemble_poisson(nodes, IEN):
    n_nodes = nodes.shape[0]
    n_elem = IEN.shape[0]
    rows, cols, vals = [], [], []
    F = np.zeros(n_nodes)
    for e in range(n_elem):
        idx = IEN[e]
        xy = nodes[idx]
        x1, y1 = xy[0]
        x2, y2 = xy[1]
        x3, y3 = xy[2]
        area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        B = np.array([
            [y2-y3, y3-y1, y1-y2],
            [x3-x2, x1-x3, x2-x1]
        ]) / (2.0 * area)
        Ke = area * (B.T @ B)
        for i in range(3):
            for j in range(3):
                rows.append(idx[i])
                cols.append(idx[j])
                vals.append(Ke[i, j])
        cx = (x1+x2+x3)/3.0
        cy = (y1+y2+y3)/3.0
        fe = (area/3.0) * f_source(cx, cy) * np.ones(3)
        for i in range(3):
            F[idx[i]] += fe[i]
    K = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    K = (K + K.T) / 2.0
    return K, F, K.nnz

def apply_dirichlet(K, F, nodes, tol=1e-12):
    fixed = []
    for i, (x, y) in enumerate(nodes):
        if abs(x) < tol or abs(x-1.0) < tol or abs(y) < tol or abs(y-1.0) < tol:
            fixed.append(i)
    free = np.setdiff1d(np.arange(len(nodes)), fixed)
    Kff = K[free, :][:, free]
    Ff = F[free]
    return Kff, Ff, free, fixed

def compute_errors(u_num, u_ex):
    max_err = np.max(np.abs(u_num - u_ex))
    l2_err = np.sqrt(np.sum((u_num - u_ex)**2) / np.sum(u_ex**2))
    return max_err, l2_err

def plot_solutions(nodes, u_num, u_ex, nx, ny, save_name):
    npx, npy = nx+1, ny+1
    X = nodes[:,0].reshape(npx, npy)
    Y = nodes[:,1].reshape(npx, npy)
    U_num = u_num.reshape(npx, npy)
    U_ex = u_ex.reshape(npx, npy)
    U_err = np.abs(U_num - U_ex)
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    cf1 = axes[0].contourf(X, Y, U_num, levels=20, cmap='viridis')
    axes[0].set_title("Numerical")
    plt.colorbar(cf1, ax=axes[0])
    cf2 = axes[1].contourf(X, Y, U_ex, levels=20, cmap='viridis')
    axes[1].set_title("Exact")
    plt.colorbar(cf2, ax=axes[1])
    cf3 = axes[2].contourf(X, Y, U_err, levels=20, cmap='hot')
    axes[2].set_title("Absolute error")
    plt.colorbar(cf3, ax=axes[2])
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()

def solve_poisson(nx, ny, solver_name="pardiso"):
    print(f"\n{'='*70}")
    print(f"算例4：二维 Poisson 方程 (T3单元)  网格 {nx}×{ny}")
    print(f"求解器: {solver_name}")
    print('='*70)

    t_gen = time.perf_counter()
    nodes, IEN = generate_triangle_mesh(nx, ny)
    t_gen = time.perf_counter() - t_gen
    n_nodes = nodes.shape[0]
    n_elem = IEN.shape[0]

    t_ass = time.perf_counter()
    K, F, nnz = assemble_poisson(nodes, IEN)
    t_ass = time.perf_counter() - t_ass

    print(f"单元类型: 线性三角形单元 (T3)")                         # 要求1
    print(f"节点数 = {n_nodes}, 单元数 = {n_elem}, 未知自由度 = {n_nodes}")
    print(f"非零元个数 = {nnz}, 稀疏度 = {100*nnz/(n_nodes*n_nodes):.4f}%")
    print(f"网格生成时间 = {t_gen:.4f}s, 装配时间 = {t_ass:.4f}s")  # 要求2

    t_bc = time.perf_counter()
    Kff, Ff, free, fixed = apply_dirichlet(K, F, nodes)
    t_bc = time.perf_counter() - t_bc
    print(f"边界处理时间 = {t_bc:.4f}s, 自由度数 = {len(free)}")

    u_free, t_solve, used_solver = solve_sparse(Kff, Ff, solver_name)
    print(f"求解时间 = {t_solve:.4f}s")                            # 要求2
    print(f"实际使用求解器 = {used_solver}")                       # 要求3

    u = np.zeros(n_nodes)
    u[free] = u_free
    u_ex = np.array([u_exact(x, y) for (x, y) in nodes])

    # 缩减系统相对残差（正确，很小）
    r_free = Ff - Kff @ u_free
    rel_res = np.linalg.norm(r_free) / np.linalg.norm(Ff)
    print(f"相对残差 (缩减系统) = {rel_res:.4e}")                  # 要求4

    max_err, l2_err = compute_errors(u, u_ex)
    print(f"节点最大误差 = {max_err:.4e}")                         # 要求5
    print(f"离散L2相对误差 = {l2_err:.4e}")                        # 要求6

    # 输出理论解与数值解的几个样本点
    print("\n理论解与数值解对比 (样本点):")
    sample_points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)]
    for (x, y) in sample_points:
        dists = (nodes[:,0]-x)**2 + (nodes[:,1]-y)**2
        idx = np.argmin(dists)
        u_num_val = u[idx]
        u_ex_val = u_exact(x, y)
        print(f"  点({x},{y}): 数值解={u_num_val:.6f}, 精确解={u_ex_val:.6f}, 误差={abs(u_num_val-u_ex_val):.2e}")

    total_time = t_gen + t_ass + t_bc + t_solve
    print(f"\n总时间 = {total_time:.4f}s")

    # 绘图（数值解云图和误差云图）满足要求7和8
    plot_solutions(nodes, u, u_ex, nx, ny, f"poisson_{nx}x{ny}.png")

if __name__ == "__main__":
    for nx in [50, 100, 200]:
        solve_poisson(nx, nx, solver_name="pardiso")