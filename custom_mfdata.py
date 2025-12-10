import numpy as np
from .springback_loader import load_vertices, load_faces_and_rebounds, face_centroids, load_pointcloud, face_rebound_to_vertex_rebound, build_common_points

class CustomMfData:
    """
    把仿真（低 fidelity）和实验（高 fidelity）包装成 MFNN 要求的数据结构。

    设计思路：
    - 把每个空间采样点（例如实验点云中的点或面心）视为一个 query location.
    - X 输入 = [point_x, point_y, point_z, coef_1, coef_2, ...]  # coef_* 是补偿参数（设计变量）
      这样 MFNN 可以学会，在某个点 (x,y,z) 上，补偿系数如何影响回弹量。
    - 如果补偿系数是全局单一值，coef_dim=1；若分区补偿则 coef_dim=k。
    - 对于训练数据，需要有多组不同 coef 下的仿真/实验结果。如果你只有一组仿真和一组实验数据（仅在一个 coef 下），那么 MFBO 无法在 design 空间上泛化（需要多组不同 coef 的数据 or 能在线运行仿真/实验作查询）。
    """

    def __init__(self, sim_vertices_csv, sim_faces_csv, exp_pointcloud_csv, coef_list_sim, coef_list_exp, coef_dim=1, query_on='exp'):
        """
        Args:
            sim_vertices_csv: 仿真顶点 csv
            sim_faces_csv: 仿真 faces + rebound csv
            exp_pointcloud_csv: 实验点云 csv
            coef_list_sim: list of coefficient vectors for each sim run, shape (Nsims, coef_dim)
            coef_list_exp: list of coefficient vectors for each experiment run, shape (Nexp, coef_dim)
            coef_dim: 1 或者 >1
            query_on: which spatial points to use as common points: 'exp' or 'sim' or 'merge'
        """
        vertices = load_vertices(sim_vertices_csv)
        faces, face_rebounds = load_faces_and_rebounds(sim_faces_csv)
        sim_centroids = face_centroids(vertices, faces)

        exp_pts, exp_rebounds = load_pointcloud(exp_pointcloud_csv)

        self.coef_dim = coef_dim

        # 建立公共查询点集合（通常使用实验点云）
        self.query_pts = build_common_points(sim_centroids, exp_pts, method=query_on)  # shape (Nq,3)

        # 构建训练集（低 fidelity = 仿真 runs；高 fidelity = 实验 runs）
        # For each sim run i, coef = coef_list_sim[i], we need sim rebound values at query_pts.
        # If sim runs are provided only on mesh face centroids, we need把 face rebound 插值/最近邻到 query_pts.
        # 为简洁起见，这里假设仿真只有一组，并且我们将 face rebound 最近邻映射到 query_pts.
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1).fit(sim_centroids)
        _, idx = nbrs.kneighbors(self.query_pts)
        sim_rebound_on_query = face_rebounds[idx.flatten()]

        # build X/y for each sim coef run
        self.Xtrain_list = []
        self.Ytrain_list = []

        for coef in coef_list_sim:
            # X shape (Nq, 3 + coef_dim)
            coef_mat = np.tile(coef.reshape(1, -1), (self.query_pts.shape[0], 1))
            X = np.hstack([self.query_pts, coef_mat])
            y = sim_rebound_on_query.reshape(-1,1)
            self.Xtrain_list.append(X)
            self.Ytrain_list.append(y)

        # experiment runs: similarly，需要把实验 re-bound 与 query_pts 对齐（如果 query_pts 就是实验点，则直接使用）
        # 若 exp runs 有多个 coef 下的实验，需分别传入 coef_list_exp 对应的 rebound 集。
        # 这里假设传入的 exp csv 是某一次实验（单个 run）
        exp_nbrs = NearestNeighbors(n_neighbors=1).fit(exp_pts)
        _, idx_e = exp_nbrs.kneighbors(self.query_pts)
        exp_reb_on_query = exp_rebounds[idx_e.flatten()]

        for coef in coef_list_exp:
            coef_mat = np.tile(coef.reshape(1, -1), (self.query_pts.shape[0], 1))
            X = np.hstack([self.query_pts, coef_mat])
            y = exp_reb_on_query.reshape(-1,1)
            self.Xtrain_list.append(X)
            self.Ytrain_list.append(y)

        # meta
        self.Nfid = 2  # 0: sim low-fid, 1: exp high-fid
        self.dim = 3 + self.coef_dim  # 输入维度：空间坐标 + coef 参数
        # bounds: spatial bounds + coef bounds (你需要设定 coef 的搜索区间)
        self.lb = np.min(self.Xtrain_list[0], axis=0)
        self.ub = np.max(self.Xtrain_list[0], axis=0)

        # Xtest / Ytest: 可以留为单个查询点集合用于测试
        self.Xtest_list = [self.Xtrain_list[0]]
        self.Ytest_list = [self.Ytrain_list[0]]
