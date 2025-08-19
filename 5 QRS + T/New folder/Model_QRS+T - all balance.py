import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, hamming_loss, accuracy_score, roc_auc_score, roc_curve, auc
)
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier

# -------------------- Configuration --------------------
modname = "mi_QRS+T_1000perclass"
subclasses = ['IMI', 'AMI', 'LMI', 'ASMI', 'ILMI', 'ALMI']
class_names_single = ['NORM'] + subclasses

# -------------------- Utility Functions --------------------
def save_model(model, model_name):
    os.makedirs("models", exist_ok=True)
    path = f"models/{model_name}.pkl"
    joblib.dump(model, path)
    return path

def convert_to_labels(y_matrix):
    labels = []
    for row in y_matrix:
        if row[0] == 0:
            labels.append('NORM')
        else:
            # find first subclass flagged as 1
            for i, sub in enumerate(subclasses, start=1):
                if row[i] == 1:
                    labels.append(sub)
                    break
            else:
                labels.append('UNKNOWN')
    return labels

def plot_roc(y_true, y_score, names, outpath):
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -------------------- Load & Sample Data --------------------
df = pd.read_csv("mi_fuzzy_features_qrs_twave.csv")
df['is_mi'] = (df['label'] != 'NORM').astype(int)
for sub in subclasses:
    df[sub] = (df['label'] == sub).astype(int)

# sample exactly 1000 per class
frames = []
for lab in class_names_single:
    subdf = df[df['label'] == lab].sample(n=1000, random_state=42)
    frames.append(subdf)
df_small = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)

print("Total samples:", len(df_small))
print("Per-class counts:\n", df_small['label'].value_counts())

# -------------------- Features & Labels --------------------
feature_cols = [
    c for c in df_small.columns
    if c not in ['beat_id','patient_id','record_id','label','is_mi'] + subclasses
       and "st_elevation" not in c.lower() and "st_depression" not in c.lower()
]
X = df_small[feature_cols]
y = df_small[['is_mi'] + subclasses]

# -------------------- Train-Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y['is_mi'], random_state=42
)
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# -------------------- Train Model --------------------
model = ClassifierChain(
    base_estimator=XGBClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        use_label_encoder=False
    ),
    order='random', random_state=42
)
model.fit(X_train, y_train)

# -------------------- Evaluate on Test Set --------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("\n— Multilabel Metrics —")
print("Hamming loss:", hamming_loss(y_test.values, y_pred))
print("Subset accuracy:", accuracy_score(y_test.values, y_pred))
print("Macro F1:", f1_score(y_test.values, y_pred, average='macro'))
print("Micro F1:", f1_score(y_test.values, y_pred, average='micro'))

# ROC & AUC
plot_roc(y_test.values, np.vstack(y_proba), ['is_mi']+subclasses, f"roc_{modname}.png")
print("ROC AUC per class:")
for i, name in enumerate(['is_mi'] + subclasses):
    auc_score = roc_auc_score(y_test.values[:, i], np.vstack(y_proba)[:, i])
    print(f"  {name}: {auc_score:.3f}")

# -------------------- Confusion Matrix (single-label) --------------------
y_true_lbl = convert_to_labels(y_test.values)
y_pred_lbl = convert_to_labels(y_pred)
cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=class_names_single)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names_single)
disp.plot(xticks_rotation='vertical')
plt.title("Confusion Matrix (single-label)")
plt.tight_layout()
plt.savefig(f"cm_{modname}.png")
plt.close()

print("\nClassification report (single-label):")
print(classification_report(y_true_lbl, y_pred_lbl, labels=class_names_single))

# -------------------- Save Final Model --------------------
model_path = save_model(model, modname)
print(f"\nModel saved to: {model_path}")

Total samples: 7000
Per-class counts:
 label
ALMI    1000
AMI     1000
NORM    1000
ASMI    1000
LMI     1000
ILMI    1000
IMI     1000
Name: count, dtype: int64
Train samples: 5600, Test samples: 1400

— Multilabel Metrics —
Hamming loss: 0.07653061224489796
Subset accuracy: 0.7307142857142858
Macro F1: 0.753704575637543
Micro F1: 0.846751123825092
ROC AUC per class:
  is_mi: 0.952
  IMI: 0.886
  AMI: 0.954
  LMI: 0.981
  ASMI: 0.850
  ILMI: 0.968
  ALMI: 0.970

Classification report (single-label):
              precision    recall  f1-score   support

        NORM       0.87      0.66      0.75       200
         IMI       0.73      0.45      0.56       205
         AMI       0.85      0.81      0.83       227
         LMI       0.80      0.90      0.85       207
        ASMI       0.56      0.55      0.56       184
        ILMI       0.80      0.77      0.79       197
        ALMI       0.58      0.98      0.73       180

   micro avg       0.73      0.73      0.73      1400
   macro avg       0.74      0.73      0.72      1400
weighted avg       0.75      0.73      0.73      1400