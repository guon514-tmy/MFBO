"""
完整工程级管线：PCA降维 + 多保真联合建模 + 概率约束贝叶斯优化
包含两套实现：
  1) sklearn 版本（基于 sklearn GaussianProcessRegressor 的多保真自回归实现与采集）
  2) BoTorch 版本（使用 PyTorch/GPyTorch/BoTorch，带采集优化）

说明：
- 数据格式（你需要准备或放置的文件）：
  * 仿真数据：放到一个文件夹（sim_dir），每组仿真为一对文件：
      - verts_XXX.csv : 顶点列表，每行 x,y,z （无表头）
      - faces_XXX.csv : 面列表，每行 v1,v2,v3,rebound_value （顶点序号从0或1开始，见下）
    文件名中的 XXX 是仿真编号或描述。脚本会扫描 sim_dir 下所有 verts_*.csv 与 faces_*.csv 并按匹配对读入。
    注意：所有仿真应使用相同拓扑（相同顶点顺序与面索引）。如果拓扑不同，请先在外部统一网格或插值到统一采样点（脚本不自动重网格）。

  * 实验数据（高保真）：一个 CSV 文件 exp_points.csv，列为 x,y,z,rebound（无表头），为现场扫描的点云。

- 本脚本完成：
  1) 将面片值（face rebound）转换为顶点值（vertex rebound）：对每个顶点取邻接面的平均。
  2) 将每次仿真（或实验）处理成同样维度的顶点场向量（长度 D = num_vertices）。
  3) 对仿真场做标准化 + PCA，得到 K 个主成分和 N_sim x K 的 scores（低维输出）。
  4) 多保真自回归建模（sklearn 版）：对每个主成分拟合 gp_sim（在仿真点上）和 gp_delta（在实验点上），并估计 rho。
  5) 用概率约束 EI 进行采集（候选点由拉丁超立方取样）。

- 注意与假设：
  * 假设网格拓扑在仿真文件间一致（顶点个数与面索引一致）。若不一致，需要提前做点云到统一网格的插值或重新网格化。
  * 实验点云未必和顶点完全对应。脚本会把每个实验点投影到最近的顶点，并用该顶点的场值近似对应的回弹值（你可以改成更复杂的双线性/重心插值）。

- 依赖（sklearn 版本）： numpy, scipy, scikit-learn
- 额外依赖（BoTorch 版本）： torch, gpytorch, botorch

使用方法示例（命令行）：
    python full_pipeline_sklearn_botorch.py --sim_dir ./sims --exp_file ./exp_points.csv --mode sklearn
或
    python full_pipeline_sklearn_botorch.py --sim_dir ./sims --exp_file ./exp_points.csv --mode botorch

输出：
- 建议的下一个实验参数 x_next（补偿系数），以及该点关键点满足概率阈值的估计。

-------------------------------------------------------------------------------
代码实现（较长，注意阅读注释）：
"""

import os
import glob
import argparse
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.spatial import cKDTree
from scipy.stats import norm

# Optional imports for BoTorch path
try:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_model
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.acquisition import qExpectedImprovement
    from botorch.optim import optimize_acqf
    BOTORCH_AVAILABLE = True
except Exception:
    BOTORCH_AVAILABLE = False

# ------------------ I/O and preprocessing helpers ------------------

def read_vertices_csv(path):
    # expects no header, columns x,y,z
    return np.loadtxt(path, delimiter=',')

def read_faces_csv(path, zero_indexed=True):
    # expects columns v1,v2,v3,rebound
    A = np.loadtxt(path, delimiter=',')
    if A.ndim == 1:
        A = A.reshape(1, -1)
    faces = A[:, :3].astype(int)
    if not zero_indexed:
        faces = faces - 1
    rebounds = A[:, 3]
    return faces, rebounds


def face_to_vertex_values(num_vertices, faces, face_values):
    """
    把面值平均分配到顶点：每个顶点取其邻接面值的均值
    """
    sums = np.zeros(num_vertices, dtype=float)
    counts = np.zeros(num_vertices, dtype=int)
    for fi, f in enumerate(faces):
        val = face_values[fi]
        for v in f:
            sums[v] += val
            counts[v] += 1
    # 防止孤立顶点
    counts[counts == 0] = 1
    return sums / counts


def load_simulation_pairs(sim_dir):
    """
    扫描 sim_dir 下 verts_*.csv 与 faces_*.csv，按匹配基名载入
    返回：
      - vertices (N_sim x num_vertices x 3) (list)
      - vertex_fields (N_sim x num_vertices) (list)
      - X_sim placeholders (这里我们假设补偿系数向量由文件名或单独提供；脚本返回 None，需用户补充或脚本扩展解析)

    注意：本函数假设所有 verts_*.csv 拥有相同的顶点数量与索引体系
    """
    verts_files = sorted(glob.glob(os.path.join(sim_dir, 'verts_*.csv')))
    faces_files = sorted(glob.glob(os.path.join(sim_dir, 'faces_*.csv')))
    # 尝试按后缀中的编号匹配（更稳健的匹配逻辑可替换）
    sims = []
    for vf in verts_files:
        base = os.path.splitext(os.path.basename(vf))[0].split('verts_')[-1]
        # find corresponding faces
        candidate = os.path.join(sim_dir, f'faces_{base}.csv')
        if not os.path.exists(candidate):
            print(f'Warning: no faces file for {vf}, expected {candidate}, skipping')
            continue
        verts = read_vertices_csv(vf)
        faces, face_vals = read_faces_csv(candidate)
        num_v = verts.shape[0]
        vertex_vals = face_to_vertex_values(num_v, faces, face_vals)
        sims.append({'verts': verts, 'vertex_vals': vertex_vals, 'id': base})
    return sims


def load_experiment_pointcloud(exp_file):
    # expects x,y,z,rebound columns
    P = np.loadtxt(exp_file, delimiter=',')
    if P.ndim == 1:
        P = P.reshape(1, -1)
    coords = P[:, :3]
    values = P[:, 3]
    return coords, values

# ------------------ PCA pipeline ------------------

def build_field_matrix_from_sims(sims):
    # sims: list of dict with 'vertex_vals'
    Y = np.vstack([s['vertex_vals'] for s in sims])  # shape (N_sim, D)
    return Y

# ------------------ map experiment points -> vertex grid ------------------

def map_exp_points_to_vertices(exp_coords, verts_reference):
    """
    把实验点云投影到参考顶点集合上（最近邻），得到每个实验点对应的顶点索引
    verts_reference: (D,3)
    返回 indices (len(exp_coords),)
    """
    tree = cKDTree(verts_reference)
    dists, idx = tree.query(exp_coords)
    return idx

# ------------------ 多保真自回归建模（sklearn版本） ------------------

def fit_multifidelity_sklearn(X_sim, Z_sim, X_exp, Z_exp):
    """
    X_sim: (N_sim, d)
    Z_sim: (N_sim, K)  主成分 scores from PCA
    X_exp: (N_exp, d)
    Z_exp: (N_exp, K)

    返回 models: list of dict per component {'gp_sim', 'rho', 'gp_delta'}
    """
    X_sim = np.asarray(X_sim)
    X_exp = np.asarray(X_exp)
    N_sim, d = X_sim.shape
    N_exp = X_exp.shape[0]
    K = Z_sim.shape[1]

    models = []
    for k in range(K):
        y_sim = Z_sim[:, k]
        y_exp = Z_exp[:, k]
        # gp on sim
        kernel_sim = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3))
        gp_sim = GaussianProcessRegressor(kernel=kernel_sim, alpha=1e-8, normalize_y=True)
        gp_sim.fit(X_sim, y_sim)
        # predict sim at exp X
        sim_mu_at_exp, sim_std = gp_sim.predict(X_exp, return_std=True)
        # estimate rho by OLS
        denom = np.dot(sim_mu_at_exp, sim_mu_at_exp)
        rho = float(np.dot(sim_mu_at_exp, y_exp) / denom) if denom != 0 else 1.0
        residuals = y_exp - rho * sim_mu_at_exp
        # fit gp on residuals
        kernel_delta = C(1.0, (1e-4, 1e4)) * RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-6)
        gp_delta = GaussianProcessRegressor(kernel=kernel_delta, alpha=1e-8, normalize_y=True)
        gp_delta.fit(X_exp, residuals)
        models.append({'gp_sim': gp_sim, 'rho': rho, 'gp_delta': gp_delta})
    return models

# ------------------ 预测高保真主成分与关键点分布 ------------------

def predict_highfid_pcs_and_keypoints(Xq, models, pca, scaler_field, keypoint_indices):
    """
    Xq: (nq, d)
    models: list of per-component models
    返回: means_kp (nq, n_key), vars_kp (nq, n_key), also pc_means (nq, K), pc_vars (nq, K)
    """
    Xq = np.asarray(Xq)
    nq = Xq.shape[0]
    K = len(models)
    pc_means = np.zeros((nq, K)); pc_vars = np.zeros((nq, K))
    for k, m in enumerate(models):
        mu_sim = m['gp_sim'].predict(Xq)
        # sklearn GP predict std
        _, sim_std = m['gp_sim'].predict(Xq, return_std=True)
        var_sim = sim_std**2
        mu_delta = m['gp_delta'].predict(Xq)
        _, delta_std = m['gp_delta'].predict(Xq, return_std=True)
        var_delta = delta_std**2
        rho = m['rho']
        pc_means[:, k] = rho * mu_sim + mu_delta
        pc_vars[:, k] = (rho**2) * var_sim + var_delta + 1e-12
    # reconstruct to field means and approx var
    Ys_means = pca.inverse_transform(pc_means)
    Y_means = scaler_field.inverse_transform(Ys_means)  # (nq, D)
    # approximate var at each field point assuming independence of PCs
    loadings = pca.components_  # (K, D)
    D = loadings.shape[1]
    n_key = len(keypoint_indices)
    means_kp = np.zeros((nq, n_key)); vars_kp = np.zeros((nq, n_key))
    for i in range(nq):
        var_field_std = np.zeros(D)
        for k in range(K):
            var_field_std += (loadings[k, :]**2) * pc_vars[i, k]
        var_field = var_field_std * (scaler_field.scale_**2)
        means_kp[i, :] = Y_means[i, keypoint_indices]
        vars_kp[i, :] = var_field[keypoint_indices]
    return pc_means, pc_vars, means_kp, vars_kp

# ------------------ 采集函数（概率约束下的 EI） ------------------

def expected_improvement(mu, sigma, fbest):
    sigma = np.maximum(sigma, 1e-12)
    z = (fbest - mu) / sigma
    return sigma * (z * norm.cdf(z) + norm.pdf(z))

def prob_feasible(means_kp, vars_kp, thresholds):
    stds = np.sqrt(np.maximum(vars_kp, 1e-12))
    z = (np.array(thresholds) - means_kp) / stds
    p_each = norm.cdf(z)
    p_tot = np.prod(p_each, axis=1)
    return p_tot

# ------------------ 简单候选生成（LHS） ------------------

def lhs_samples(n, bounds, random_state=None):
    rng = np.random.default_rng(random_state)
    d = len(bounds)
    seg = np.linspace(0, 1, n+1)
    X = np.zeros((n, d))
    for i in range(d):
        pts = seg[:-1] + rng.random(n) / n
        rng.shuffle(pts)
        lb, ub = bounds[i]
        X[:, i] = lb + pts * (ub - lb)
    return X

# ------------------ 主流程（sklearn 版本） ------------------

def run_pipeline_sklearn(sim_dir, exp_file, bounds, keypoint_indices, thresholds, alpha=0.05, n_cand=200):
    sims = load_simulation_pairs(sim_dir)
    if len(sims) == 0:
        raise RuntimeError('No simulation pairs found in sim_dir')
    # reference verts
    verts_ref = sims[0]['verts']
    D = verts_ref.shape[0]
    # build Y_sim
    Y_sim = build_field_matrix_from_sims(sims)  # (N_sim, D)
    N_sim = Y_sim.shape[0]
    print(f'Loaded {N_sim} simulations, field dim D={D}')
    # load exp pointcloud and map to vertex grid
    exp_coords, exp_vals = load_experiment_pointcloud(exp_file)
    idxs = map_exp_points_to_vertices(exp_coords, verts_ref)
    # For each unique experimental run we need the full field vector across vertices.
    # Here we assume exp_vals are measurements at scattered points and we approximate the experiment's full-field
    # by interpolating/assigning measured values to corresponding vertices and then filling missing vertices by e.g. nearest.
    # For simplicity, we build one high-fidelity field by setting vertex value = mean of exp measurements that map to that vertex;
    # if no measurement maps to a vertex, we set NaN and later fill by nearest neighbor.
    vertex_exp_vals = np.full(D, np.nan)
    counts = np.zeros(D, dtype=int)
    sums = np.zeros(D)
    for i, vid in enumerate(idxs):
        sums[vid] += exp_vals[i]
        counts[vid] += 1
    mask = counts > 0
    vertex_exp_vals[mask] = sums[mask] / counts[mask]
    # fill NaN by nearest vertex that has data
    known_idx = np.where(mask)[0]
    if len(known_idx) == 0:
        raise RuntimeError('No experiment points map to any vertex; check exp_file or meshes')
    tree_known = cKDTree(verts_ref[known_idx])
    nan_idx = np.where(~mask)[0]
    if len(nan_idx) > 0:
        dists, nn = tree_known.query(verts_ref[nan_idx])
        vertex_exp_vals[nan_idx] = vertex_exp_vals[known_idx[nn]]
    # Now vertex_exp_vals is a single high-fidelity field (one experiment). If you have multiple experiments
    # you should provide them as multiple exp files or combine differently.
    Y_exp = vertex_exp_vals.reshape(1, -1)
    # PCA on Y_sim
    scaler_field = StandardScaler()
    Ys = scaler_field.fit_transform(Y_sim)
    pca = PCA(n_components=min(Ys.shape[0], Ys.shape[1]))
    pca.fit(Ys)
    cum = np.cumsum(pca.explained_variance_ratio_)
    K = int(np.searchsorted(cum, 0.995) + 1)
    print(f'Choosing K={K} principal components (cumulative var {cum[K-1]:.4f})')
    pca = PCA(n_components=K)
    Z_sim = pca.fit_transform(Ys)
    # project experiment field to PC space
    Yexp_std = scaler_field.transform(Y_exp)
    Z_exp = pca.transform(Yexp_std)
    # Build X_sim and X_exp: we need the design variables of simulations and experiments (补偿系数)
    # This script does not parse design variables from file names. For demonstration, we assume
    # user provides X_sim.csv and X_exp.csv with rows matching sims order and experiment respectively.
    X_sim_file = os.path.join(sim_dir, 'X_sim.csv')
    X_exp_file = os.path.join(sim_dir, 'X_exp.csv')
    if not os.path.exists(X_sim_file) or not os.path.exists(X_exp_file):
        raise RuntimeError('Expected X_sim.csv and X_exp.csv in sim_dir describing design variables for sims and experiment')
    X_sim = np.loadtxt(X_sim_file, delimiter=',')
    X_exp = np.loadtxt(X_exp_file, delimiter=',')
    # Fit multi-fidelity models per PC
    models = fit_multifidelity_sklearn(X_sim, Z_sim, X_exp, Z_exp)
    # Candidate generation
    X_cand = lhs_samples(n_cand, bounds)
    # compute current best fbest from experiment
    # objective choose: mean absolute keypoint value
    def obj_from_field(Y):
        return np.mean(np.abs(Y[:, keypoint_indices]), axis=1)
    fbest = obj_from_field(Y_exp).min()
    pc_means, pc_vars, means_kp, vars_kp = predict_highfid_pcs_and_keypoints(X_cand, models, pca, scaler_field, keypoint_indices)
    pf = prob_feasible(means_kp, vars_kp, thresholds)
    # compute proxy objective mean/sigma
    weights = np.ones(len(keypoint_indices)) / len(keypoint_indices)
    mu_obj = np.sum(weights * np.abs(means_kp), axis=1)
    sigma_obj = np.sqrt(np.sum((weights**2) * vars_kp, axis=1))
    ei = expected_improvement(mu_obj, sigma_obj, fbest)
    feasible_mask = pf >= (1 - alpha)
    scores = ei * feasible_mask
    best_idx = int(np.argmax(scores))
    x_next = X_cand[best_idx]
    print('SKLEARN suggestion:')
    print('x_next =', x_next)
    print('feasible_probability =', pf[best_idx])
    return x_next, pf[best_idx]

# ------------------ BoTorch 版本（简化实现：对每个 PC 建立 GP on residuals） ------------------

def run_pipeline_botorch(sim_dir, exp_file, bounds, keypoint_indices, thresholds, alpha=0.05, n_cand=200):
    if not BOTORCH_AVAILABLE:
        raise RuntimeError('BoTorch stack not available. Install torch gpytorch botorch')
    sims = load_simulation_pairs(sim_dir)
    if len(sims) == 0:
        raise RuntimeError('No simulation pairs found')
    verts_ref = sims[0]['verts']
    D = verts_ref.shape[0]
    Y_sim = build_field_matrix_from_sims(sims)
    exp_coords, exp_vals = load_experiment_pointcloud(exp_file)
    idxs = map_exp_points_to_vertices(exp_coords, verts_ref)
    # build single exp field as before
    vertex_exp_vals = np.full(D, np.nan)
    counts = np.zeros(D, dtype=int); sums = np.zeros(D)
    for i, vid in enumerate(idxs):
        sums[vid] += exp_vals[i]; counts[vid] += 1
    mask = counts > 0
    vertex_exp_vals[mask] = sums[mask] / counts[mask]
    known_idx = np.where(mask)[0]
    tree_known = cKDTree(verts_ref[known_idx])
    nan_idx = np.where(~mask)[0]
    if len(known_idx) == 0:
        raise RuntimeError('No experiment points map to any vertex')
    if len(nan_idx) > 0:
        dists, nn = tree_known.query(verts_ref[nan_idx]); vertex_exp_vals[nan_idx] = vertex_exp_vals[known_idx[nn]]
    Y_exp = vertex_exp_vals.reshape(1, -1)
    # PCA
    scaler_field = StandardScaler(); Ys = scaler_field.fit_transform(Y_sim)
    pca = PCA(n_components=min(Ys.shape[0], Ys.shape[1])); pca.fit(Ys)
    cum = np.cumsum(pca.explained_variance_ratio_); K = int(np.searchsorted(cum, 0.995) + 1)
    pca = PCA(n_components=K); Z_sim = pca.fit_transform(Ys)
    Yexp_std = scaler_field.transform(Y_exp); Z_exp = pca.transform(Yexp_std)
    # load X_sim, X_exp
    X_sim_file = os.path.join(sim_dir, 'X_sim.csv'); X_exp_file = os.path.join(sim_dir, 'X_exp.csv')
    if not os.path.exists(X_sim_file) or not os.path.exists(X_exp_file):
        raise RuntimeError('Expected X_sim.csv and X_exp.csv in sim_dir for design variables')
    X_sim = np.loadtxt(X_sim_file, delimiter=','); X_exp = np.loadtxt(X_exp_file, delimiter=',')
    # Fit sklearn gp_sim to get predictions at X_exp, then residuals to fit BoTorch SingleTaskGP on residuals
    # We'll reuse fit_multifidelity_sklearn to get gp_sim and rho, then build torch GP on residuals
    models = fit_multifidelity_sklearn(X_sim, Z_sim, X_exp, Z_exp)  # returns gp_sim, rho, gp_delta (sklearn)
    # Prepare torch datasets for residual GP per PC
    train_x = torch.tensor(X_exp, dtype=torch.float32)
    next_x = torch.tensor(lhs_samples(n_cand, bounds), dtype=torch.float32)
    # For each PC, build a SingleTaskGP on residuals (we could instead use multitask, but separation is simpler)
    pcs_gp = []
    for k, m in enumerate(models):
        # compute residuals at X_exp
        sim_mu_at_exp = m['gp_sim'].predict(X_exp)
        rho = m['rho']
        resid = (Z_exp[:, k] - rho * sim_mu_at_exp).reshape(-1, 1)
        train_y = torch.tensor(resid, dtype=torch.float32)
        gp = SingleTaskGP(train_X=train_x, train_Y=train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        pcs_gp.append({'gp': gp, 'rho': rho, 'sim_gp': m['gp_sim']})
    # For each candidate, predict pc means/vars by combining sim prediction (from sklearn gp_sim) and torch gp pred for delta
    X_cand = next_x.numpy()
    K = len(pcs_gp)
    pc_means = np.zeros((X_cand.shape[0], K)); pc_vars = np.zeros_like(pc_means)
    for k, item in enumerate(pcs_gp):
        sim_mu = item['sim_gp'].predict(X_cand)
        # torch gp predict
        gp = item['gp']
        with torch.no_grad():
            post = gp.posterior(torch.tensor(X_cand, dtype=torch.float32))
            delta_mu = post.mean.numpy().reshape(-1)
            delta_var = post.variance.numpy().reshape(-1)
        rho = item['rho']
        pc_means[:, k] = rho * sim_mu + delta_mu
        pc_vars[:, k] = (rho**2) * (item['sim_gp'].predict(X_cand, return_std=True)[1]**2) + delta_var
    # reconstruct keypoint means/vars
    Ys_means = pca.inverse_transform(pc_means); Y_means = scaler_field.inverse_transform(Ys_means)
    loadings = pca.components_
    n_key = len(keypoint_indices)
    means_kp = np.zeros((X_cand.shape[0], n_key)); vars_kp = np.zeros_like(means_kp)
    for i in range(X_cand.shape[0]):
        var_field_std = np.zeros(loadings.shape[1])
        for k in range(K):
            var_field_std += (loadings[k, :]**2) * pc_vars[i, k]
        var_field = var_field_std * (scaler_field.scale_**2)
        means_kp[i, :] = Y_means[i, keypoint_indices]
        vars_kp[i, :] = var_field[keypoint_indices]
    pf = prob_feasible(means_kp, vars_kp, thresholds)
    # For BoTorch acquisition we will construct a proxy objective: weighted mean abs keypoints reconstructed from pc_means
    weights = np.ones(n_key) / n_key
    mu_obj = np.sum(weights * np.abs(means_kp), axis=1)
    sigma_obj = np.sqrt(np.sum((weights**2) * vars_kp, axis=1))
    fbest = mu_obj.min()
    # Use simple EI with feasibility masking
    ei = expected_improvement(mu_obj, sigma_obj, fbest)
    scores = ei * (pf >= (1 - alpha))
    best_idx = int(np.argmax(scores))
    x_next = X_cand[best_idx]
    print('BoTorch suggestion:')
    print('x_next =', x_next)
    print('feasible_probability =', pf[best_idx])
    return x_next, pf[best_idx]

# ------------------ CLI ------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_dir', required=True, help='simulation directory with verts_*.csv, faces_*.csv, and X_sim.csv, X_exp.csv')
    parser.add_argument('--exp_file', required=True, help='experiment pointcloud csv with x,y,z,value')
    parser.add_argument('--mode', default='sklearn', choices=['sklearn', 'botorch'])
    parser.add_argument('--bounds', nargs='+', type=float, default=None,
                        help='bounds flattened: lb1 ub1 lb2 ub2 ...')
    parser.add_argument('--keypoints', nargs='+', type=int, required=True, help='indices of keypoint vertices')
    parser.add_argument('--thresholds', nargs='+', type=float, required=True, help='thresholds for keypoints')
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()

    if args.bounds is None:
        raise RuntimeError('Please provide bounds for each design variable as lb ub ...')
    b = np.array(args.bounds).reshape(-1, 2)
    if args.mode == 'sklearn':
        run_pipeline_sklearn(args.sim_dir, args.exp_file, b, args.keypoints, args.thresholds, alpha=args.alpha)
    else:
        run_pipeline_botorch(args.sim_dir, args.exp_file, b, args.keypoints, args.thresholds, alpha=args.alpha)

# End of file
