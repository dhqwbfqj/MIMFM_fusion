import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# =========================
# 核心处理函数
# =========================
def process_excel_files(folder_path, show_confusion_matrix=False, random_state=None):
    """
    处理指定文件夹中的所有Excel文件，使用XGBoost分类器进行5折交叉验证。
    返回每个文件的指标字典列表（包含每折指标）
    """
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    if not excel_files:
        print(f"在{folder_path}中未找到Excel文件")
        return []

    print(f"在{folder_path}中找到{len(excel_files)}个Excel文件\n")
    all_results = []

    for file_path in excel_files:
        file_name = os.path.basename(file_path)
        print(f"正在处理: {file_name}")

        try:
            data = pd.read_excel(file_path)
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values

            # 检查类别分布
            neg_count = np.sum(y == 0)
            pos_count = np.sum(y == 1)
            if pos_count == 0:
                print(f"错误: {file_name}中没有正样本")
                continue
            class_ratio = neg_count / pos_count

            print(f"数据集信息: {X.shape[0]}个样本, {X.shape[1]}个特征")
            print(f"类别分布: {neg_count}负, {pos_count}正 (比例: {class_ratio:.2f})")

            # 5折交叉验证
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            metrics_dict = {
                'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'uar': [], 'auc': []
            }
            total_conf_matrix = np.zeros((2, 2))

            for train_idx, test_idx in kf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # 标准化
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # XGBoost分类器（保留原参数）
                clf = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='auc',
                    random_state=42,
                    colsample_bytree=0.8,
                    learning_rate=0.05,
                    max_depth=3,
                    n_estimators=300,
                    subsample=0.8,
                )
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred_prob = clf.predict_proba(X_test)[:, 1]

                # 计算指标
                metrics_dict['accuracy'].append(accuracy_score(y_test, y_pred))
                metrics_dict['precision'].append(precision_score(y_test, y_pred))
                metrics_dict['recall'].append(recall_score(y_test, y_pred))
                metrics_dict['f1'].append(f1_score(y_test, y_pred))
                metrics_dict['auc'].append(roc_auc_score(y_test, y_pred_prob))

                # UAR
                cm = confusion_matrix(y_test, y_pred, labels=[0,1])
                recall_neg = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1])>0 else 0
                recall_pos = cm[1,1]/(cm[1,0]+cm[1,1]) if (cm[1,0]+cm[1,1])>0 else 0
                metrics_dict['uar'].append((recall_neg+recall_pos)/2)

                total_conf_matrix += cm

            # 保存结果
            category = file_name.replace('.xlsx','')
            all_results.append({
                'category': category,
                'file_name': file_name,
                **metrics_dict
            })

            if show_confusion_matrix:
                plt.figure(figsize=(6,5))
                sns.heatmap(total_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
                plt.xlabel('预测标签')
                plt.ylabel('真实标签')
                plt.title(f'混淆矩阵 - {file_name}')
                plt.show()

            print("-"*50)

        except Exception as e:
            print(f"处理 {file_name} 出错: {e}")

    return all_results


# =========================
# 多次重复实验
# =========================
def repeat_experiments(folder_path, repeats=5, show_confusion_matrix=False):
    """
    对指定文件夹数据进行多次重复实验
    返回: DataFrame，每个文件每个指标为“均值±标准差”
    """
    repeat_results = []

    for i in range(repeats):
        print(f"\n===== 第 {i+1} 次重复实验 =====")
        results = process_excel_files(folder_path, show_confusion_matrix=show_confusion_matrix, random_state=None)
        repeat_results.append(results)

    # 指标列表
    metrics = ['accuracy','f1','precision','recall','uar','auc']
    summary = {}

    # 假设每次文件顺序一致
    for idx, file_result in enumerate(repeat_results[0]):
        category = file_result['category']
        summary[category] = {}

        for metric in metrics:
            # 收集该文件该指标的所有重复实验值
            values = []
            for run in repeat_results:
                for r in run:
                    if r['category']==category:
                        values.extend(r[metric])
            mean = np.mean(values)*100
            std = np.std(values)*100
            summary[category][metric] = f"{mean:.2f}±{std:.2f}"

    df_summary = pd.DataFrame.from_dict(summary, orient='index')
    df_summary.index.name = '文件名'
    return df_summary


# =========================
# 使用示例
# =========================
if __name__ == "__main__":
    folder_path = r"E:\data_gen\face\6_new_face_mcca\gcca"  # 修改为你的数据路径
    repeats = 5  # 重复实验次数
    show_confusion_matrix = False  # 是否显示混淆矩阵

    df_summary = repeat_experiments(folder_path, repeats=repeats, show_confusion_matrix=show_confusion_matrix)
    print("\n===== 重复实验结果汇总 =====")
    print(df_summary)

    # 保存结果
    df_summary.to_excel(os.path.join(folder_path, f"重复实验结果汇总_{repeats}次.xlsx"))