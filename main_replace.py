# 在 main.run() 中，替换/扩展 domain 分支：
    if domain == 'springback':
        # 从 args 读取文件路径（你需要在 argparse 中添加这些参数）
        sim_vertices_csv = args.sim_vertices
        sim_faces_csv = args.sim_faces
        exp_pointcloud_csv = args.exp_pc
        # coef_list_sim / coef_list_exp 必须提供：每一条仿真/实验 run 对应的系数向量
        # 如果你只有一组数据，则长度为1；若你有多次仿真结果请传入多行系数
        coef_list_sim = ...  # e.g., np.array([[0.0],[0.1],[0.2]])
        coef_list_exp = ...  # e.g., np.array([[0.0]])
        from data.custom_mfdata import CustomMfData
        SynData = CustomMfData(sim_vertices_csv, sim_faces_csv, exp_pointcloud_csv, coef_list_sim, coef_list_exp, coef_dim=coef_list_sim.shape[1], query_on='exp')
        Nfid = SynData.Nfid
        costs = [1, 10]  # 你可以根据实验/仿真的相对成本调整
