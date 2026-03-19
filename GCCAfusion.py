import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =========================
# GCCA
# =========================
def gcca(views, k=10, reg=1e-4):
    """
    参数：
        views: list of [n_samples, d_i]
        k: 共享子空间维度
        reg: 正则项

    返回：
        G: [n_samples, k]
        Ws: list of [d_i, k]
    """

    V = len(views)
    n = views[0].shape[0]

    # ===== a. 标准化 =====
    views = [StandardScaler().fit_transform(X) for X in views]

    # ===== b. 构建 C =====
    C = np.zeros((n, n))
    projections = []

    for X in views:
        XtX = X.T @ X
        d = XtX.shape[0]

        # 正则化防止不可逆
        inv = np.linalg.inv(XtX + reg * np.eye(d))

        # 投影矩阵
        P = X @ inv @ X.T
        C += P

        projections.append((X, inv))

    # ===== c. 特征值分解 =====
    eigvals, eigvecs = np.linalg.eigh(C)

    # 按特征值降序排序
    idx = np.argsort(-eigvals)
    eigvecs = eigvecs[:, idx]

    # ===== d. 取前 k 个 =====
    G = eigvecs[:, :k]

    # ===== e. 求每个视图的 W_i =====
    Ws = []
    for X, inv in projections:
        W = inv @ X.T @ G   # [d_i, k]
        Ws.append(W)

    return G, Ws


# =========================
# 批量融合 xlsx
# =========================
def fuse_xlsx_gcca(folder_path, output_folder, k=10, output_prefix='gcca'):
    views = []
    labels = None
    filenames = []

    # ===== 读取数据 =====
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_excel(file_path)

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            views.append(X)
            filenames.append(filename)

            if labels is None:
                labels = y
            else:
                assert np.array_equal(labels, y), f"标签不一致：{filename}"

    print(f"[INFO] 共 {len(views)} 个视图, {len(labels)} 个样本")

    # ===== GCCA =====
    G, Ws = gcca(views, k=k)

    # ===== 保存融合特征 =====
    fused_df = pd.DataFrame(G)
    fused_df['label'] = labels

    os.makedirs(output_folder, exist_ok=True)

    fused_file = os.path.join(output_folder, f"{output_prefix}_k{k}.xlsx")
    fused_df.to_excel(fused_file, index=False)

    print(f"[SUCCESS] 融合特征保存至: {fused_file}")

    # ===== 保存每个视图的 W_i =====
    for i, W in enumerate(Ws):
        W_df = pd.DataFrame(W)
        W_file = os.path.join(output_folder, f"{output_prefix}_W_view{i+1}_k{k}.xlsx")
        W_df.to_excel(W_file, index=False)

    print(f"[SUCCESS] 投影矩阵 W_i 已全部保存")

    return fused_df, Ws


# =========================
# 新样本映射函数（可选，但强烈推荐）
# =========================
def project_new_sample(X_new, W):
    """
    X_new: [m, d_i]
    W: [d_i, k]
    return: [m, k]
    """
    return X_new @ W


# =========================
# 使用示例
# =========================
if __name__ == "__main__":

    folder_path = r"E:\data_gen\face\4_new_face_label\FaceNet\area4"
    output_folder = r"E:\data_gen\face\6_new_face_mcca\gcca"

    fuse_xlsx_gcca(folder_path, output_folder, k=40)