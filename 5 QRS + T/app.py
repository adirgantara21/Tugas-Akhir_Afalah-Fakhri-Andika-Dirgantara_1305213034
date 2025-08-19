import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
MODEL_PATH = "best_model_QRS+T.pkl"
model = joblib.load(MODEL_PATH)

# MI subclass list
mi_subclasses = ['IMI', 'AMI', 'LMI', 'ASMI', 'ILMI', 'ALMI']
class_names = ['NORM'] + mi_subclasses

def convert_to_labels(y_mat):
    labels = []
    for row in y_mat:
        if row[0] == 0:
            labels.append('NORM')
        else:
            for i, sc in enumerate(mi_subclasses):
                if row[i + 1] == 1:
                    labels.append(sc)
                    break
            else:
                labels.append('UNKNOWN')
    return np.array(labels)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    try:
        df = pd.read_csv(file)

        # Ambil kolom fitur
        drop_cols = ["beat_id", "patient_id", "record_id", "label", "is_mi"] + mi_subclasses
        feature_cols = [c for c in df.columns if c not in drop_cols and "st_" not in c.lower()]
        X = df[feature_cols].values

        # Prediksi multi-label
        y_pred = model.predict(X)
        labels = convert_to_labels(y_pred)

        # Gabungkan dengan data asli
        df_result = df.copy()
        df_result["Predicted_Label"] = labels

        # Hitung jumlah prediksi tiap kelas
        class_counts = df_result["Predicted_Label"].value_counts().to_dict()

        return render_template(
            "index.html",
            tables=[df_result.to_html(classes='table table-striped', index=False)],
            counts=class_counts
        )

    except Exception as e:
        return f"Error saat memproses: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
