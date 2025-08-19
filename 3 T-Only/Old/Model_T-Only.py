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

modname = "mi_T" #<--------------------------------------------------------------------------------------
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
    and "qrs" not in col.lower() #<--------------------------------------------------------------------------------------
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

# ===================== Contoh Penggunaan Model =====================
print("\nContoh: Prediksi dengan model yang dimuat kembali")

try:
    loaded_model = load_model(model_filename)
    sample_data = X_test.iloc[:5]
    predictions = loaded_model.predict(sample_data)
    pred_labels = convert_to_labels(predictions)
    true_labels = convert_to_labels(y_test.iloc[:5])

    print("\nSample Predictions vs Actual:")
    for i, (pred, actual) in enumerate(zip(pred_labels, true_labels)):
        status = "✅ BENAR" if pred == actual else "❌ SALAH"
        print(f"- Sample {i+1}: Predicted = {pred:<5} | Actual = {actual:<5} --> {status}")
except Exception as e:
    print(f"Gagal memuat model: {e}")

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
Model disimpan sebagai: models/mi_T.pkl

Classification Report:
              precision    recall  f1-score   support

        NORM       0.61      0.87      0.72      4671
         IMI       0.70      0.42      0.52      4715
         AMI       0.37      0.17      0.23       648
         PMI       1.00      0.20      0.33        30
         LMI       0.33      0.21      0.26       277
        ASMI       0.59      0.63      0.61      3757
        ILMI       0.24      0.40      0.30       801
        ALMI       0.73      0.21      0.33       327
        IPMI       1.00      0.09      0.16        56
       IPLMI       0.72      0.13      0.22        99

    accuracy                           0.58     15381
   macro avg       0.63      0.33      0.37     15381
weighted avg       0.60      0.58      0.57     15381


Confusion Matrix:
Actual \ Pred NORM     IMI     AMI     PMI     LMI    ASMI    ILMI    ALMI    IPMI    IPLMI
NORM         4053     233     31       0      23      234     97       0       0       0
IMI          1362    1970     77       0      37      836     423      6       0       4
AMI           252     64      112      0      11      142     66       1       0       0
PMI           10       4       0       6       1       3       6       0       0       0
LMI           93      12       4       0      58      74      35       1       0       0
ASMI          695     308     56       0      40     2381     261     16       0       0
ILMI          95      160     15       0       2      208     318      2       0       1   
ALMI          52      25       9       0       2      105     64      70       0       0
IPMI           9      14       1       0       2      10      15       0       5       0
IPLMI         19      24       1       0       0      21      21       0       0      13

============================================================
OVERALL METRICS
============================================================
- Accuracy: 91.7%
- Precision: 62.9%
- Recall: 33.3%
- Specificity: 94.5%
- F1: 37.0%

============================================================
CLASS-WISE PERFORMANCE
============================================================

NORM:
  - Accuracy    : 79.2%
  - Precision   : 61.0%
  - Recall      : 86.8%
  - Specificity : 75.8%
  - F1          : 71.7%

IMI:
  - Accuracy    : 76.7%
  - Precision   : 70.0%
  - Recall      : 41.8%
  - Specificity : 92.1%
  - F1          : 52.3%

AMI:
  - Accuracy    : 95.3%
  - Precision   : 36.6%
  - Recall      : 17.3%
  - Specificity : 98.7%
  - F1          : 23.5%

PMI:
  - Accuracy    : 99.8%
  - Precision   : 100.0%
  - Recall      : 20.0%
  - Specificity : 100.0%
  - F1          : 33.3%

LMI:
  - Accuracy    : 97.8%
  - Precision   : 33.0%
  - Recall      : 20.9%
  - Specificity : 99.2%
  - F1          : 25.6%

ASMI:
  - Accuracy    : 80.4%
  - Precision   : 59.3%
  - Recall      : 63.4%
  - Specificity : 86.0%
  - F1          : 61.3%

ILMI:
  - Accuracy    : 90.4%
  - Precision   : 24.3%
  - Recall      : 39.7%
  - Specificity : 93.2%
  - F1          : 30.2%

ALMI:
  - Accuracy    : 98.2%
  - Precision   : 72.9%
  - Recall      : 21.4%
  - Specificity : 99.8%
  - F1          : 33.1%

IPMI:
  - Accuracy    : 99.7%
  - Precision   : 100.0%
  - Recall      : 8.9%
  - Specificity : 100.0%
  - F1          : 16.4%

IPLMI:
  - Accuracy    : 99.4%
  - Precision   : 72.2%
  - Recall      : 13.1%
  - Specificity : 100.0%
  - F1          : 22.2%

Contoh: Prediksi dengan model yang dimuat kembali
Model berhasil dimuat dari: models/mi_T.pkl

- Sample 1: Predicted = IMI   | Actual = IMI   --> ✅ BENAR
- Sample 2: Predicted = ASMI  | Actual = ASMI  --> ✅ BENAR
- Sample 3: Predicted = IMI   | Actual = NORM  --> ❌ SALAH
- Sample 4: Predicted = ILMI  | Actual = ILMI  --> ✅ BENAR
- Sample 5: Predicted = ILMI  | Actual = ASMI  --> ❌ SALAH