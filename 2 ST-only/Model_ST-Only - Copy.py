import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# ========= Load & Preprocessing =========
df = pd.read_csv("mi_fuzzy_features_qrs_st_twave.csv")
df['is_mi'] = (df['label'] != 'NORM').astype(int)
mi_subclasses = ['IMI', 'AMI', 'LMI', 'ASMI', 'ILMI', 'ALMI']
for sc in mi_subclasses:
    df[sc] = (df['label'] == sc).astype(int)

feature_cols = [c for c in df.columns if c not in ["beat_id","patient_id","record_id","label","is_mi"] + mi_subclasses
                and "qrs" not in c.lower()
                and "t_inversion" not in c.lower()
                and "t_amplitude" not in c.lower() #<--------------------------------------------------------------------------------------
                ]
X = df[feature_cols].values
y = df[["is_mi"] + mi_subclasses].values
class_names = ['NORM'] + mi_subclasses

# Undersampling
df_all = pd.concat([df[feature_cols], df[["is_mi"] + mi_subclasses], df[['label']]], axis=1)
mi_data = df_all[df_all['is_mi'] == 1]
norm_data = df_all[df_all['is_mi'] == 0]
max_mi = int(mi_data[mi_subclasses].sum().max())
unders = norm_data.sample(n=max_mi, random_state=42)
df_bal = pd.concat([mi_data, unders]).sample(frac=1, random_state=42)
X = df_bal[feature_cols].values
y = df_bal[["is_mi"] + mi_subclasses].values

# ========= Helpers =========
def convert_to_labels(y_mat):
    labels = []
    for row in y_mat:
        if row[0] == 0:
            labels.append('NORM')
        else:
            for i, sc in enumerate(mi_subclasses):
                if row[i+1] == 1:
                    labels.append(sc)
                    break
            else:
                labels.append('UNKNOWN')
    return np.array(labels)

def specificity(cm, idx):
    tp = cm[idx, idx]
    fn = cm[idx].sum() - tp
    fp = cm[:, idx].sum() - tp
    tn = cm.sum() - (tp+fn+fp)
    return tn/(tn+fp) if (tn+fp)>0 else 0

# ========= CV Setup =========
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_auc = -np.inf

# Aggregasi for visuals
agg_class_metrics = {cls: {'accuracy': [], 'precision': [], 'recall': [], 'specificity': [], 'f1': []} for cls in class_names}
agg_cm = np.zeros((len(class_names), len(class_names)), dtype=float)
all_auc = {cls: [] for cls in class_names}
interp_tprs = {cls: [] for cls in class_names}

# Create model dir
os.makedirs('models', exist_ok=True)

# ========= Cross‑Validation =========
for fold, (tr, te) in enumerate(kf.split(X, y[:,0]), start=1):
    print(f"--- Fold {fold} ---")
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    model = ClassifierChain(
        base_estimator=XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss'),
        order='random', random_state=42
    )
    model.fit(X_tr, y_tr)

    # Predict & compute per-class AUC
    y_pred = model.predict(X_te)
    y_score = model.predict_proba(X_te)
    if isinstance(y_score, list):
        y_score = np.vstack([p[:,1] for p in y_score]).T

    # Compute fold average AUC
    fold_aucs = [roc_auc_score(y_te[:,idx], y_score[:,idx]) for idx in range(len(class_names))]
    mean_fold_auc = np.mean(fold_aucs)
    print(f"Fold {fold} mean AUC: {mean_fold_auc:.4f}")

    # Save best model only
    if mean_fold_auc > best_auc:
        best_auc = mean_fold_auc
        joblib.dump(model, 'models/best_model_ST.pkl') #<----------------------------------------
        print(f">> New best model saved with AUC={best_auc:.4f}")

    # Aggregate confusion matrix & class metrics
    y_true_lab = convert_to_labels(y_te)
    y_pred_lab = convert_to_labels(y_pred)
    cm = confusion_matrix(y_true_lab, y_pred_lab, labels=class_names)
    agg_cm += cm
    for idx, cls in enumerate(class_names):
        tp = cm[idx, idx]; fn = cm[idx].sum() - tp; fp = cm[:, idx].sum() - tp; tn = cm.sum() - (tp+fn+fp)
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        acc = (tp+tn)/cm.sum()
        spec = specificity(cm, idx)
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        agg_class_metrics[cls]['accuracy'].append(acc)
        agg_class_metrics[cls]['precision'].append(prec)
        agg_class_metrics[cls]['recall'].append(rec)
        agg_class_metrics[cls]['specificity'].append(spec)
        agg_class_metrics[cls]['f1'].append(f1)

    # Aggregate ROC data
    mean_fpr = np.linspace(0,1,100)
    for idx, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_te[:,idx], y_score[:,idx])
        all_auc[cls].append(roc_auc_score(y_te[:,idx], y_score[:,idx]))
        interp_tprs[cls].append(np.interp(mean_fpr, fpr, tpr))

# ========= Compute Means =========
mean_class_metrics = {cls: {m: np.mean(vals)*100 for m, vals in mets.items()} for cls, mets in agg_class_metrics.items()}
raw_df = pd.DataFrame(mean_class_metrics).T[['accuracy','precision','recall','specificity','f1']]
raw_df['Overall'] = raw_df.mean(axis=1)

# Print class-wise and overall metrics
print("\n=== Class-wise Metrics (mean over 5 folds) ===")
print(raw_df[['accuracy','precision','recall','specificity','f1']].round(1).astype(str) + ' %')
print("\n=== Overall Metrics ===")
print(raw_df['Overall'].mean().round(1).astype(str) + ' %')

# ========= Visualization Blocks =========
# 1. Class‑wise Metrics Table
df_metrics = raw_df.drop(columns=['Overall']).round(1).astype(str) + ' %'
fig, ax = plt.subplots(figsize=(8,3))
ax.axis('tight'); ax.axis('off')
tbl = ax.table(cellText=df_metrics.values, rowLabels=df_metrics.index,
               colLabels=df_metrics.columns, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1,2)
ax.set_title('Class‑wise Metrics', pad=20, fontweight='bold')
plt.tight_layout(); plt.show()

# 2. Confusion Matrix Rata‑rata
avg_cm_int = np.rint(agg_cm/kf.get_n_splits()).astype(int)
plt.figure(figsize=(8,6)); sns.heatmap(avg_cm_int, annot=True, fmt='d',
            xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix'); plt.tight_layout(); plt.show()

# 3. Average ROC Curves
plt.figure(figsize=(10,8)); mean_fpr = np.linspace(0,1,100)
for cls in class_names:
    plt.plot(mean_fpr, np.mean(interp_tprs[cls],axis=0),
             label=f"{cls}: AUC={np.mean(all_auc[cls]):.2f}")
plt.plot([0,1],[0,1],'k--'); plt.title('Average ROC Curves per Class')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend(loc='lower right'); plt.grid(True); plt.tight_layout(); plt.show()
