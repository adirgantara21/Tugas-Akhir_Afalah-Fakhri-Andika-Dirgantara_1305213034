from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, roc_curve
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib, os

# ========== Setup dan Load Data ==========
df = pd.read_csv("mi_fuzzy_features_qrs_twave.csv")
df['is_mi'] = (df['label'] != 'NORM').astype(int)
mi_subclasses = ['IMI', 'AMI', 'LMI', 'ASMI', 'ILMI', 'ALMI']
for subclass in mi_subclasses:
    df[subclass] = (df['label'] == subclass).astype(int)

feature_cols = [
    col for col in df.columns 
    if col not in ["beat_id", "patient_id", "record_id", "label", "is_mi"] + mi_subclasses
    and "st_elevation" not in col.lower()
    and "st_depression" not in col.lower()
]

# ========== Sampling Spesifik: NORM, IMI, ASMI ==========
df_all = pd.concat([df[feature_cols], df[['is_mi'] + mi_subclasses], df[['label']]], axis=1)
sampled_list = []
targets = ['NORM', 'IMI', 'ASMI']
for lbl, grp in df_all.groupby('label'):
    if lbl in targets:
        # pastikan hanya 5000 sampel
        sampled = grp.sample(n=5000, random_state=42)
    else:
        sampled = grp.copy()
    sampled_list.append(sampled)

# Satukan dan acak
df_balanced = pd.concat(sampled_list).sample(frac=1, random_state=42)

X = df_balanced[feature_cols].values
y = df_balanced[['is_mi'] + mi_subclasses].values
labels_str = df_balanced['label'].values

# ========== Stratified K-Fold (5 splits -> 80/20) ==========
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_metrics = []

# helper untuk konversi matrix multioutput ke label string
def convert_to_labels(y_matrix):
    labels = []
    for row in y_matrix:
        if row[0] == 0:
            labels.append('NORM')
        else:
            for i, subclass in enumerate(mi_subclasses):
                if row[i + 1] == 1:
                    labels.append(subclass)
                    break
            else:
                labels.append('UNKNOWN')
    return labels

fold = 1
for train_idx, test_idx in kf.split(X, y[:, 0]):
    print(f"\n=== Fold {fold} ===")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = ClassifierChain(
        base_estimator=XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss"
        ),
        order='random',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # simpan model per fold
    os.makedirs('models', exist_ok=True)
    model_path = f"models/mi_QRS+T_fold{fold}.pkl"
    joblib.dump(model, model_path)

    # Predict & Probabilities
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)
    if isinstance(y_score, list):
        y_score = np.vstack([p[:, 1] for p in y_score]).T

    # Hitung report
    y_test_labels = convert_to_labels(y_test)
    y_pred_labels = convert_to_labels(y_pred)
    class_names = ['NORM'] + mi_subclasses
    filtered = [(yt, yp) for yt, yp in zip(y_test_labels, y_pred_labels) if yt in class_names]
    y_true_filt, y_pred_filt = zip(*filtered)

    print(classification_report(y_true_filt, y_pred_filt, labels=class_names, zero_division=0))

    # Confusion matrix & metrics
    cm = confusion_matrix(y_true_filt, y_pred_filt, labels=class_names)
    def calculate_specificity(cm, idx):
        tp = cm[idx, idx]
        fn = cm[idx].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    cls_metrics = {}
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = calculate_specificity(cm, i)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = (tp + tn) / cm.sum()
        cls_metrics[cls] = {
            'accuracy': acc, 'precision': precision,
            'recall': recall, 'specificity': specificity, 'f1': f1
        }

    avg_metrics = {m: np.mean([v[m] for v in cls_metrics.values()]) for m in cls_metrics['NORM']}
    all_metrics.append(avg_metrics)

    # AUC per kelas
    auc_scores = {}
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
        auc_scores[cls] = roc_auc_score(y_test[:, i], y_score[:, i])
    print("AUC per class:", auc_scores)

    fold += 1

# Rata-rata keseluruhan 5-fold
avg_df = pd.DataFrame(all_metrics).mean().round(4) * 100
print("\n==== RATA-RATA METRIK DARI 5-FOLD ====")
for key, val in avg_df.items():
    print(f"- {key.title():<12}: {val:.2f}%")

=== Fold 1 ===
              precision    recall  f1-score   support

        NORM       0.90      0.79      0.84      1000
         IMI       0.84      0.67      0.75       996
         AMI       0.90      0.91      0.90       624
         LMI       0.90      0.90      0.90       292
        ASMI       0.71      0.87      0.78       972
        ILMI       0.87      0.90      0.89       798
        ALMI       0.81      0.88      0.84       352

   micro avg       0.83      0.83      0.83      5034
   macro avg       0.85      0.84      0.84      5034
weighted avg       0.84      0.83      0.83      5034

AUC per class: {'NORM': np.float64(0.9724594213126323), 'IMI': np.float64(0.9053643481092407), 'AMI': np.float64(0.9912013100364084), 'LMI': np.float64(0.9692402356860503), 'ASMI': np.float64(0.9335421962171464), 'ILMI': np.float64(0.9853220520422998), 'ALMI': np.float64(0.9769039136001781)}

=== Fold 2 ===
              precision    recall  f1-score   support

        NORM       0.90      0.78      0.83      1000
         IMI       0.85      0.65      0.74       997
         AMI       0.90      0.91      0.90       589
         LMI       0.90      0.89      0.89       289
        ASMI       0.71      0.89      0.79      1019
        ILMI       0.86      0.87      0.87       794
        ALMI       0.81      0.90      0.86       364

   micro avg       0.83      0.82      0.83      5052
   macro avg       0.85      0.84      0.84      5052
weighted avg       0.84      0.82      0.83      5052

AUC per class: {'NORM': np.float64(0.9724701176470589), 'IMI': np.float64(0.9089068286448811), 'AMI': np.float64(0.9856745767082926), 'LMI': np.float64(0.9831439553778991), 'ASMI': np.float64(0.9433974758482708), 'ILMI': np.float64(0.9790210126215919), 'ALMI': np.float64(0.9802446325675961)}

=== Fold 3 ===
              precision    recall  f1-score   support

        NORM       0.91      0.81      0.86      1000
         IMI       0.82      0.66      0.73       967
         AMI       0.93      0.89      0.91       626
         LMI       0.91      0.89      0.90       289
        ASMI       0.73      0.88      0.80      1000
        ILMI       0.89      0.90      0.89       829
        ALMI       0.80      0.93      0.86       321

   micro avg       0.84      0.83      0.84      5032
   macro avg       0.86      0.85      0.85      5032
weighted avg       0.85      0.83      0.84      5032

AUC per class: {'NORM': np.float64(0.9791008235294116), 'IMI': np.float64(0.9157313937572389), 'AMI': np.float64(0.9869115988812364), 'LMI': np.float64(0.9780425728990626), 'ASMI': np.float64(0.9355929411764706), 'ILMI': np.float64(0.9879752000609003), 'ALMI': np.float64(0.9809848130051086)}

=== Fold 4 ===
              precision    recall  f1-score   support

        NORM       0.89      0.74      0.81      1000
         IMI       0.83      0.66      0.73      1038
         AMI       0.90      0.91      0.91       662
         LMI       0.91      0.93      0.92       279
        ASMI       0.71      0.88      0.78       972
        ILMI       0.88      0.88      0.88       778
        ALMI       0.79      0.91      0.85       324

   micro avg       0.83      0.82      0.82      5053
   macro avg       0.84      0.84      0.84      5053
weighted avg       0.84      0.82      0.82      5053

AUC per class: {'NORM': np.float64(0.9636125882352943), 'IMI': np.float64(0.9112490782368753), 'AMI': np.float64(0.988261773126796), 'LMI': np.float64(0.9834452008026482), 'ASMI': np.float64(0.9399783464832033), 'ILMI': np.float64(0.9864196416663984), 'ALMI': np.float64(0.9761306847516077)}

=== Fold 5 ===
              precision    recall  f1-score   support

        NORM       0.91      0.78      0.84      1000
         IMI       0.84      0.67      0.75      1002
         AMI       0.91      0.91      0.91       612
         LMI       0.93      0.90      0.91       268
        ASMI       0.72      0.89      0.80      1037
        ILMI       0.87      0.89      0.88       778
        ALMI       0.81      0.91      0.85       349

   micro avg       0.84      0.83      0.83      5046
   macro avg       0.86      0.85      0.85      5046
weighted avg       0.84      0.83      0.83      5046

AUC per class: {'NORM': np.float64(0.9712679999999999), 'IMI': np.float64(0.9140184790494341), 'AMI': np.float64(0.9859863954206088), 'LMI': np.float64(0.9778074201453592), 'ASMI': np.float64(0.9393730339645324), 'ILMI': np.float64(0.988239591908062), 'ALMI': np.float64(0.9807401448391622)}

==== RATA-RATA METRIK DARI 5-FOLD ====
- Accuracy    : 95.28%
- Precision   : 85.00%
- Recall      : 85.55%
- Specificity : 97.15%
- F1          : 84.91%