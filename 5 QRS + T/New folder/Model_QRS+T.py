from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.utils import resample

modname = "mi_QRS+T"
# ===================== Fungsi Save dan Load Model =====================
def save_model(model, model_name="mi_classifier"):
    if not os.path.exists('models'):
        os.makedirs('models')
    filename = f"models/{model_name}.pkl"
    joblib.dump(model, filename)
    print(f"Model disimpan sebagai: {filename}")
    return filename

def load_model(filename):
    if os.path.exists(filename):
        model = joblib.load(filename)
        print(f"Model berhasil dimuat dari: {filename}")
        return model
    else:
        raise FileNotFoundError(f"File model {filename} tidak ditemukan")

# ===================== Load Data =====================
df = pd.read_csv("mi_fuzzy_features_all_leads_ST-QRS_FIX.csv")
df['is_mi'] = (df['label'] != 'NORM').astype(int)
mi_subclasses = ['IMI', 'AMI', 'PMI', 'LMI', 'ASMI', 'ILMI', 'ALMI', 'IPMI','IPLMI']
for subclass in mi_subclasses:
    df[subclass] = (df['label'] == subclass).astype(int)

# Fitur dan target
feature_cols = [
    col for col in df.columns 
    if col not in ["beat_id", "patient_id", "record_id", "label", "is_mi"] + mi_subclasses
    and "st_elevation" not in col.lower()
    and "st_depression" not in col.lower()
    ]
X_all = df[feature_cols]
y_all = df[["is_mi"] + mi_subclasses]

# Gabungkan agar mudah diproses
df_all = pd.concat([X_all, y_all, df[['label']]], axis=1)

# ===================== Undersampling SEBELUM Split =====================
mi_data = df_all[df_all['is_mi'] == 1]
norm_data = df_all[df_all['is_mi'] == 0]

# Hitung subclass MI terbanyak
mi_counts = mi_data[mi_subclasses].sum()
max_mi_class_size = int(mi_counts.max())

print("\nJumlah masing-masing subclass MI:")
print(mi_counts)
print(f"Jumlah MI terbesar: {max_mi_class_size}")

# Undersample NORM
undersampled_norm = resample(
    norm_data,
    replace=False,
    n_samples=max_mi_class_size,
    random_state=42
)

# Gabungkan MI dan NORM hasil undersampling
df_balanced = pd.concat([mi_data, undersampled_norm])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Split ulang
X = df_balanced[feature_cols]
y = df_balanced[["is_mi"] + mi_subclasses]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y["is_mi"], random_state=42
)

print("\nSetelah undersampling dan split:")
print(f"- Total MI: {len(mi_data)}")
print(f"- Total NORM (undersampled): {len(undersampled_norm)}")
print(f"- Total data: {len(df_balanced)}")
print(f"- Train size: {len(X_train)}, Test size: {len(X_test)}")

# ===================== Latih Model =====================
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

print("\nMelatih model...")
model.fit(X_train, y_train)
print("Pelatihan selesai.")

model_filename = save_model(model, modname)

# ===================== Evaluasi =====================
y_pred = model.predict(X_test)

def convert_to_labels(y_vector):
    labels = []
    if isinstance(y_vector, pd.DataFrame):
        y_vector = y_vector.values
    for row in y_vector:
        if row[0] == 0:
            labels.append('NORM')
        else:
            found = False
            for i, subclass in enumerate(mi_subclasses):
                if i+1 < len(row) and row[i+1] == 1:
                    labels.append(subclass)
                    found = True
                    break
            if not found:
                labels.append('UNKNOWN')
    return labels

y_test_labels = convert_to_labels(y_test)
y_pred_labels = convert_to_labels(y_pred)

class_names = ['NORM'] + mi_subclasses
filtered_indices = [i for i, label in enumerate(y_test_labels) if label in class_names]
y_test_labels_filtered = [y_test_labels[i] for i in filtered_indices]
y_pred_labels_filtered = [y_pred_labels[i] for i in filtered_indices]

print("\nClassification Report:")
print(classification_report(
    y_test_labels_filtered, 
    y_pred_labels_filtered,
    labels=class_names,
    zero_division=0
))

cm = confusion_matrix(y_test_labels_filtered, y_pred_labels_filtered, labels=class_names)
print("\nConfusion Matrix:")
print("Actual \\ Pred".ljust(12) + " ".join(f"{label:^7}" for label in class_names))
for i, row in enumerate(cm):
    print(f"{class_names[i]:<12}" + " ".join(f"{val:^7}" for val in row))

# ===================== Hitung Metrik =====================
def calculate_specificity(cm, class_idx):
    tp = cm[class_idx, class_idx]
    fn = np.sum(cm[class_idx, :]) - tp
    fp = np.sum(cm[:, class_idx]) - tp
    tn = np.sum(cm) - (tp + fn + fp)
    return tn / (tn + fp) if (tn + fp) > 0 else 0

class_metrics = {}
total_samples = len(y_test_labels_filtered)

for i, class_name in enumerate(class_names):
    tp = cm[i, i]
    fn = np.sum(cm[i, :]) - tp
    fp = np.sum(cm[:, i]) - tp
    tn = total_samples - (tp + fn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = calculate_specificity(cm, i)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total_samples
    class_metrics[class_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1
    }

macro_metrics = {m: np.mean([v[m] for v in class_metrics.values()]) for m in ['accuracy', 'precision', 'recall', 'specificity', 'f1']}

print("\n" + "="*60)
print("OVERALL METRICS")
print("="*60)
for key, value in macro_metrics.items():
    print(f"- {key.title()}: {value*100:.1f}%")

print("\n" + "="*60)
print("CLASS-WISE PERFORMANCE")
print("="*60)
for class_name in class_names:
    m = class_metrics[class_name]
    print(f"\n{class_name}:")
    for key in m:
        print(f"  - {key.title():<12}: {m[key]*100:.1f}%")

# ===================== Simpan Class-Wise Performance sebagai Gambar =====================
metrics_df = pd.DataFrame(class_metrics).T
metrics_df = (metrics_df * 100).round(1)  # Konversi ke persen dan dibulatkan

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
tbl = ax.table(
    cellText=metrics_df.values,
    colLabels=metrics_df.columns.str.title(),
    rowLabels=metrics_df.index,
    loc='center',
    cellLoc='center',
    rowLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.2, 1.2)
plt.title("Class-wise Performance (%)", fontsize=12)
plt.tight_layout()

# ===================== Confusion Matrix Plot =====================
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()

from sklearn.metrics import roc_auc_score, roc_curve

# ROC Curve and AUC per class
y_test_bin = y_test.values
y_score = model.predict_proba(X_test)

# model.predict_proba() mengembalikan list of arrays, satu per label
# Kita perlu menyatukannya kembali ke array (n_samples, n_labels)
if isinstance(y_score, list):
    y_score = np.vstack([p[:, 1] for p in y_score]).T  # ambil probabilitas kelas positif

fig, ax = plt.subplots(figsize=(10, 8))
auc_scores = {}

for i, class_name in enumerate(y.columns):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    auc = roc_auc_score(y_test_bin[:, i], y_score[:, i])
    auc_scores[class_name] = auc
    ax.plot(fpr, tpr, label=f"{class_name} (AUC = {auc:.2f})")

ax.plot([0, 1], [0, 1], 'k--', label="Random")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve per Class")
ax.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# Cetak skor AUC secara tekstual
print("\n" + "="*60)
print("AUC SCORES PER CLASS")
print("="*60)
for label, auc in auc_scores.items():
    print(f"- {label:<10}: {auc:.3f}")

Jumlah masing-masing subclass MI:
IMI      23354
AMI       3057
PMI        184
LMI       1364
ASMI     19133
ILMI      3915
ALMI      1720
IPMI       322
IPLMI      498
dtype: int64
Jumlah MI terbesar: 23354

Setelah undersampling dan split:
- Total MI: 53547
- Total NORM (undersampled): 23354
- Total data: 76901
- Train size: 61520, Test size: 15381

Melatih model...
Pelatihan selesai.
Model disimpan sebagai: models/mi_QRS+T.pkl

Classification Report:
              precision    recall  f1-score   support

        NORM       0.79      0.91      0.85      4671
         IMI       0.86      0.73      0.79      4715
         AMI       0.80      0.64      0.71       648
         PMI       1.00      0.60      0.75        30
         LMI       0.89      0.55      0.68       277
        ASMI       0.81      0.86      0.83      3757
        ILMI       0.58      0.70      0.63       801
        ALMI       0.95      0.65      0.77       327
        IPMI       1.00      0.54      0.70        56
       IPLMI       1.00      0.58      0.73        99

    accuracy                           0.80     15381
   macro avg       0.87      0.67      0.74     15381
weighted avg       0.81      0.80      0.80     15381


Confusion Matrix:
Actual \ Pred NORM     IMI     AMI     PMI     LMI    ASMI    ILMI    ALMI    IPMI    IPLMI
NORM         4259     221     25       0       6      114     43       3       0       0
IMI           633    3437     33       0       3      430     177      2       0       0
AMI           128     29      416      0       1      48      26       0       0       0
PMI            2       2       0      18       1       1       6       0       0       0
LMI           53       8       2       0      153     41      19       1       0       0
ASMI          239     150     32       0       6     3232     92       6       0       0
ILMI          28      128      7       0       0      79      559      0       0       0
ALMI          19       7       3       0       1      61      25      211      0       0
IPMI          10       7       2       0       0       2       5       0      30       0
IPLMI         12      14       0       0       0       4      12       0       0      57

============================================================
OVERALL METRICS
============================================================
- Accuracy: 96.1%
- Precision: 86.8%
- Recall: 67.5%
- Specificity: 97.4%
- F1: 74.4%

============================================================
CLASS-WISE PERFORMANCE
============================================================

NORM:
  - Accuracy    : 90.0%
  - Precision   : 79.1%
  - Recall      : 91.2%
  - Specificity : 89.5%
  - F1          : 84.7%

IMI:
  - Accuracy    : 88.0%
  - Precision   : 85.9%
  - Recall      : 72.9%
  - Specificity : 94.7%
  - F1          : 78.8%

AMI:
  - Accuracy    : 97.8%
  - Precision   : 80.0%
  - Recall      : 64.2%
  - Specificity : 99.3%
  - F1          : 71.2%

PMI:
  - Accuracy    : 99.9%
  - Precision   : 100.0%
  - Recall      : 60.0%
  - Specificity : 100.0%
  - F1          : 75.0%

LMI:
  - Accuracy    : 99.1%
  - Precision   : 89.5%
  - Recall      : 55.2%
  - Specificity : 99.9%
  - F1          : 68.3%

ASMI:
  - Accuracy    : 91.5%
  - Precision   : 80.6%
  - Recall      : 86.0%
  - Specificity : 93.3%
  - F1          : 83.2%

ILMI:
  - Accuracy    : 95.8%
  - Precision   : 58.0%
  - Recall      : 69.8%
  - Specificity : 97.2%
  - F1          : 63.3%

ALMI:
  - Accuracy    : 99.2%
  - Precision   : 94.6%
  - Recall      : 64.5%
  - Specificity : 99.9%
  - F1          : 76.7%

IPMI:
  - Accuracy    : 99.8%
  - Precision   : 100.0%
  - Recall      : 53.6%
  - Specificity : 100.0%
  - F1          : 69.8%

IPLMI:
  - Accuracy    : 99.7%
  - Precision   : 100.0%
  - Recall      : 57.6%
  - Specificity : 100.0%
  - F1          : 73.1%

============================================================
AUC SCORES PER CLASS
============================================================
- is_mi     : 0.944
- IMI       : 0.945
- AMI       : 0.939
- PMI       : 0.998
- LMI       : 0.940
- ASMI      : 0.948
- ILMI      : 0.933
- ALMI      : 0.961
- IPMI      : 0.999
- IPLMI     : 0.973