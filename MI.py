import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import seaborn as sns

# Remove Chinese font settings
plt.rcParams[
    'axes.unicode_minus'] = False  # Solve the problem of negative sign '-' showing as square when saving images
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'normal',  # 明确指定正常字重
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',
})


class InfoNCEEstimator(nn.Module):
    def __init__(self, u_dim, v_dim, hidden_dim=64, tau=0.2, scale=20.0):
        super(InfoNCEEstimator, self).__init__()

        self.tau = tau
        self.scale = scale

        # g_theta(u)
        self.g = nn.Sequential(
            nn.Linear(u_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, v_dim)
        )

    def forward(self, u, v):
        """
        u: [B, d_u]
        v: [B, d_v]
        return: InfoNCE loss
        """

        # ===== 映射 g(u) =====
        u_proj = self.g(u)  # [B, d_v]

        # ===== L2归一化（余弦相似度）=====
        u_proj = F.normalize(u_proj, dim=1)
        v = F.normalize(v, dim=1)

        # ===== 相似度矩阵 =====
        sim_matrix = torch.matmul(u_proj, v.T)  # [B, B]

        # ===== 数值稳定（可选但推荐）=====
        sim_matrix = self.scale * torch.tanh(sim_matrix / self.scale)

        # ===== InfoNCE =====
        logits = sim_matrix / self.tau
        labels = torch.arange(u.size(0)).to(u.device)

        loss = F.cross_entropy(logits, labels)

        return loss

    def encode_only(self, v, a):
        v_enc = self.video_proj(v)
        a_enc = self.audio_proj(a)
        fusion = self.fusion(torch.cat([v_enc, a_enc], dim=1))
        return self.classifier(fusion)


class MMIM_InterModalOnly(nn.Module):
    def __init__(self, v_dim, a_dim, shared_dim=32, fusion_dim=64, n_class=2, dropout=0.3):
        super().__init__()

        self.video_proj = nn.Sequential(
            nn.Linear(v_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(a_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.mi_va = InfoNCEEstimator(shared_dim, shared_dim)
        self.mi_av = InfoNCEEstimator(shared_dim, shared_dim)

        self.fusion = nn.Sequential(
            nn.Linear(shared_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(fusion_dim, n_class)

    def forward(self, v, a):
        v_enc = self.video_proj(v)
        a_enc = self.audio_proj(a)

        # 模态间 MI
        mi_va = self.mi_va(v_enc, a_enc)
        mi_av = self.mi_av(a_enc, v_enc)
        mi_loss = mi_va + mi_av

        fusion = self.fusion(torch.cat([v_enc, a_enc], dim=1))
        pred = self.classifier(fusion)

        return pred, mi_loss, {
            'mi_inter': mi_loss.item(),
            'mi_fusion': 0.0
        }

    def encode_only(self, v, a):
        v_enc = self.video_proj(v)
        a_enc = self.audio_proj(a)
        fusion = self.fusion(torch.cat([v_enc, a_enc], dim=1))
        return self.classifier(fusion)


class MMIM_FusionModalOnly(nn.Module):
    def __init__(self, v_dim, a_dim, shared_dim=32, fusion_dim=64, n_class=2, dropout=0.3):
        super().__init__()

        self.video_proj = nn.Sequential(
            nn.Linear(v_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(a_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fusion = nn.Sequential(
            nn.Linear(shared_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.mi_fv = InfoNCEEstimator(fusion_dim, shared_dim)
        self.mi_fa = InfoNCEEstimator(fusion_dim, shared_dim)

        self.classifier = nn.Linear(fusion_dim, n_class)

    def forward(self, v, a):
        v_enc = self.video_proj(v)
        a_enc = self.audio_proj(a)

        fusion = self.fusion(torch.cat([v_enc, a_enc], dim=1))

        # 融合 vs 单模态
        mi_fv = self.mi_fv(fusion, v_enc)
        mi_fa = self.mi_fa(fusion, a_enc)
        mi_loss = mi_fv + mi_fa

        pred = self.classifier(fusion)

        return pred, mi_loss, {
            'mi_inter': 0.0,
            'mi_fusion': mi_loss.item()
        }

    def encode_only(self, v, a):
        v_enc = self.video_proj(v)
        a_enc = self.audio_proj(a)
        fusion = self.fusion(torch.cat([v_enc, a_enc], dim=1))
        return self.classifier(fusion)


class MMIM_Complete(nn.Module):
    def __init__(self, v_dim, a_dim, shared_dim=32, fusion_dim=64, n_class=2, dropout=0.3):
        super().__init__()

        self.video_proj = nn.Sequential(
            nn.Linear(v_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(a_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 一级MI（模态间）
        self.mi_va = InfoNCEEstimator(shared_dim, shared_dim)
        self.mi_av = InfoNCEEstimator(shared_dim, shared_dim)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(shared_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 二级MI（融合 vs 单模态）
        self.mi_fv = InfoNCEEstimator(fusion_dim, shared_dim)
        self.mi_fa = InfoNCEEstimator(fusion_dim, shared_dim)

        self.classifier = nn.Linear(fusion_dim, n_class)

    def forward(self, v, a):
        v_enc = self.video_proj(v)
        a_enc = self.audio_proj(a)

        # ===== 一级 MI =====
        mi_va = self.mi_va(v_enc, a_enc)
        mi_av = self.mi_av(a_enc, v_enc)
        mi_inter = mi_va + mi_av

        # ===== 融合 =====
        fusion = self.fusion(torch.cat([v_enc, a_enc], dim=1))

        # ===== 二级 MI =====
        mi_fv = self.mi_fv(fusion, v_enc)
        mi_fa = self.mi_fa(fusion, a_enc)
        mi_fusion = mi_fv + mi_fa

        mi_loss = mi_inter + mi_fusion

        pred = self.classifier(fusion)

        return pred, mi_loss, {
            'mi_inter': mi_inter.item(),
            'mi_fusion': mi_fusion.item()
        }

    def encode_only(self, v, a):
        v_enc = self.video_proj(v)
        a_enc = self.audio_proj(a)
        fusion = self.fusion(torch.cat([v_enc, a_enc], dim=1))
        return self.classifier(fusion)


class VADataset(Dataset):
    def __init__(self, video_features, audio_features, labels):
        self.video_features = video_features
        self.audio_features = audio_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.video_features[idx], self.audio_features[idx], self.labels[idx]


def load_features_from_xlsx(video_path, audio_path):
    video_df = pd.read_excel(video_path)
    audio_df = pd.read_excel(audio_path)
    video_features = video_df.iloc[:, :-1].values.astype(np.float32)
    audio_features = audio_df.iloc[:, :-1].values.astype(np.float32)
    video_labels = video_df.iloc[:, -1].values
    audio_labels = audio_df.iloc[:, -1].values
    if not np.array_equal(video_labels, audio_labels):
        print("Warning: Video and audio file labels are inconsistent!")
    labels = video_labels.astype(np.int64)
    print(f"Loaded {len(labels)} samples")
    print(f"Video feature dimensions: {video_features.shape[1]}")
    print(f"Audio feature dimensions: {audio_features.shape[1]}")
    print(f"Label distribution: {np.bincount(labels)}")
    return video_features, audio_features, labels


def train_one_fold(model, train_loader, val_loader, criterion, optimizer,
                   n_epochs=40, mi_weight=0.005, device="cuda"):
    model.to(device)
    best_val_acc = 0.0
    best_state = None
    for epoch in range(n_epochs):
        model.train()
        for video_feat, audio_feat, labels in train_loader:
            video_feat, audio_feat, labels = video_feat.to(device), audio_feat.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, mi_loss, mi_details = model(video_feat, audio_feat)
            task_loss = criterion(logits, labels)
            loss = task_loss + mi_weight * mi_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for video_feat, audio_feat, labels in val_loader:
                video_feat, audio_feat = video_feat.to(device), audio_feat.to(device)
                logits = model.encode_only(video_feat, audio_feat)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
        val_acc = accuracy_score(all_labels, all_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
    model.load_state_dict(best_state)
    return model, all_preds, all_probs, all_labels


def plot_roc_curves(all_folds_results, model_name, mean_auc):
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for fold_idx in range(5):
        y_true = all_folds_results[fold_idx]['y_true']
        y_probs = all_folds_results[fold_idx]['y_probs']
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        plt.plot(fpr, tpr, color=colors[fold_idx], lw=4, alpha=0.6,  # Adjust transparency and line width
                 label=f'Fold {fold_idx + 1} (AUC = {auc:.3f})')
    all_true = np.concatenate([all_folds_results[i]['y_true'] for i in range(5)])
    all_probs = np.concatenate([all_folds_results[i]['y_probs'] for i in range(5)])
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for fold_idx in range(5):
        y_true = all_folds_results[fold_idx]['y_true']
        y_probs = all_folds_results[fold_idx]['y_probs']
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    plt.plot(mean_fpr, mean_tpr, color='black', lw=6,  # Increase the line width of the average line
             label=f'Mean (AUC = {mean_auc:.3f})', linestyle='--')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=28)  # ROC curve x-axis label font size, set to 20 if single chart
    plt.ylabel('True Positive Rate', fontsize=28)  # ROC curve y-axis label font size
    plt.legend(loc="lower right", fontsize=26)  # ROC curve legend font size
    plt.xticks(fontsize=28)  # ROC x-axis tick font size
    plt.yticks(fontsize=28)  # ROC y-axis tick font size
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=['Non-Severe OSA', 'Severe OSA'],
                     yticklabels=['Non-Severe OSA', 'Severe OSA'],
                     annot_kws={"size": 18})  # Font size of numbers in confusion matrix

    # Set colorbar tick font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)  # ←←← Key modification: Set colorbar font size

    plt.xlabel('Predicted Label', fontsize=18)  # Confusion matrix x-axis label font size
    plt.ylabel('True Label', fontsize=18)  # Confusion matrix y-axis label font size
    plt.xticks(fontsize=18)  # Confusion matrix x-axis tick font size
    plt.yticks(fontsize=18)  # Confusion matrix y-axis tick font size
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


def main_ablation():
    torch.manual_seed(42)
    np.random.seed(42)
    batch_size = 16
    n_epochs = 50
    learning_rate = 0.001
    mi_weight = 0.05
    shared_dim = 32
    fusion_dim = 64
    dropout = 0.4
    n_splits = 5
    video_path = r"E:\data_gen\new\deepspectrum_resnet\5-mcca_xlsx\fused_maxvar_mcca.xlsx"
    audio_path = r"E:\data_gen\face\6_new_face_mcca\fused_maxvar_mcca.xlsx"
    video_features, audio_features, labels = load_features_from_xlsx(video_path, audio_path)
    model_configs = [
        {
            'name': 'InterModalOnly',
            'class': MMIM_InterModalOnly,
            'description': 'Inter-Modal MI Only',
        },
        {
            'name': 'FusionModalOnly',
            'class': MMIM_FusionModalOnly,
            'description': 'Fusion Layer MI Only',
        },
        {
            'name': 'Complete',
            'class': MMIM_Complete,
            'description': 'Complete MMIM',
        }
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=32)
    print("Starting ablation experiment - 5-fold cross validation...")
    print(f"Dataset size: {len(labels)}")
    print("=" * 60)
    results_summary = []
    for config in model_configs:
        print(f"\n{'=' * 60}")
        print(f"Starting training: {config['description']}")
        print(f"{'=' * 60}")
        all_metrics = {
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'uar': [],
            'auc': []
        }
        all_folds_results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(video_features, labels)):
            print(f"\n--- Fold {fold + 1}/{n_splits} ---")
            X_train_v, X_val_v = video_features[train_idx], video_features[val_idx]
            X_train_a, X_val_a = audio_features[train_idx], audio_features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            train_dataset = VADataset(
                torch.tensor(X_train_v, dtype=torch.float32),
                torch.tensor(X_train_a, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            )
            val_dataset = VADataset(
                torch.tensor(X_val_v, dtype=torch.float32),
                torch.tensor(X_val_a, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            model = config['class'](
                v_dim=video_features.shape[1],
                a_dim=audio_features.shape[1],
                shared_dim=shared_dim,
                fusion_dim=fusion_dim,
                n_class=2,
                dropout=dropout
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            trained_model, preds, probs, true_labels = train_one_fold(
                model, train_loader, val_loader, criterion, optimizer,
                n_epochs=n_epochs,
                mi_weight=mi_weight,
                device=device
            )
            all_folds_results.append({
                'y_true': np.array(true_labels),
                'y_pred': np.array(preds),
                'y_probs': np.array(probs)
            })
            acc = accuracy_score(true_labels, preds)
            f1 = f1_score(true_labels, preds, average='binary')
            prec = precision_score(true_labels, preds, average='binary', zero_division=0)
            rec = recall_score(true_labels, preds, average='binary')
            uar = recall_score(true_labels, preds, average='macro')
            try:
                auc = roc_auc_score(true_labels, probs)
            except ValueError:
                auc = float('nan')
            all_metrics['accuracy'].append(acc)
            all_metrics['f1'].append(f1)
            all_metrics['precision'].append(prec)
            all_metrics['recall'].append(rec)
            all_metrics['uar'].append(uar)
            all_metrics['auc'].append(auc)
            print(f"Fold {fold + 1} - Accuracy: {acc:.4f}, F1: {f1:.4f}, UAR: {uar:.4f}, AUC: {auc:.4f}")
        all_true = np.concatenate([result['y_true'] for result in all_folds_results])
        all_pred = np.concatenate([result['y_pred'] for result in all_folds_results])
        all_probs = np.concatenate([result['y_probs'] for result in all_folds_results])
        mean_auc = np.nanmean(all_metrics['auc'])  # Calculate the average AUC of each fold
        print(f"\n[{config['description']}] 5-fold cross-validation results:")
        metric_results = {}
        for metric_name, values in all_metrics.items():
            values = np.array(values)
            values = values[~np.isnan(values)]
            if len(values) == 0:
                mean, std = 0.0, 0.0
            else:
                mean, std = values.mean(), values.std()
            if metric_name == 'auc':
                mean = mean_auc  # Use the average AUC of each fold
            mean_pct = mean * 100  # Convert to percentage
            std_pct = std * 100  # Convert to percentage
            print(f"{metric_name:>10}: {mean_pct:.2f}±{std_pct:.2f}")
            metric_results[metric_name] = f"{mean_pct:.2f}±{std_pct:.2f}"
        results_summary.append({
            'Model': config['description'],
            'Accuracy': metric_results['accuracy'],
            'F1': metric_results['f1'],
            'Precision': metric_results['precision'],
            'Recall': metric_results['recall'],
            'UAR': metric_results['uar'],
            'AUC': metric_results['auc']
        })
        plot_roc_curves(all_folds_results, config['name'], mean_auc)
        plot_confusion_matrix(all_true, all_pred, config['name'])
    results_df = pd.DataFrame(results_summary)
    print("\n" + "=" * 80)
    print("Ablation Experiment Results Summary Table:")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)
    output_path = "mmim_ablation_results.xlsx"
    results_df.to_excel(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    return results_df


if __name__ == "__main__":
    results_table = main_ablation()
