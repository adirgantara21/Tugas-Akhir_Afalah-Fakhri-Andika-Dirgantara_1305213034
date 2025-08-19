import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Simulasi model (pada implementasi nyata, gunakan model yang sebenarnya)
MODEL_PATH = "best_model_QRS+T.pkl"
model = joblib.load(MODEL_PATH)

# MI subclass list
mi_subclasses = ['IMI', 'AMI', 'LMI', 'ASMI', 'ILMI', 'ALMI']
class_names = ['NORM'] + mi_subclasses

# Fungsi untuk mensimulasikan prediksi model
def predict_simulation(X):
    # Simulasi: 60% NORM, 40% terdistribusi ke subclasses
    base = np.zeros((len(X), len(class_names)))
    base[:, 0] = (np.random.random(len(X)) > 0.4).astype(int)  # Kolom NORM
    
    # Untuk baris yang bukan NORM, set satu subclass secara acak
    non_norm_idx = np.where(base[:, 0] == 0)[0]
    for idx in non_norm_idx:
        subclass_idx = np.random.randint(1, len(class_names))
        base[idx, subclass_idx] = 1
    
    return base

def convert_to_labels(y_mat):
    labels = []
    for row in y_mat:
        if row[0] == 1:
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
        flash('Tidak ada file yang diunggah', 'danger')
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        flash('Tidak ada file yang dipilih', 'danger')
        return redirect(url_for('home'))
    
    try:
        # Untuk demo, kita akan gunakan data sample jika file tidak sesuai
        if not file.filename.endswith('.csv'):
            # Generate sample data for demo
            data = {
                'beat_id': range(1, 11),
                'patient_id': ['P001']*10,
                'record_id': ['R001']*10,
                'label': ['NORM', 'IMI', 'AMI', 'NORM', 'LMI', 'ASMI', 'NORM', 'ILMI', 'ALMI', 'NORM'],
                'is_mi': [0,1,1,0,1,1,0,1,1,0],
                'f1': np.random.normal(0.5, 0.1, 10),
                'f2': np.random.normal(0.3, 0.05, 10),
                'f3': np.random.normal(0.7, 0.15, 10),
                'f4': np.random.normal(0.2, 0.03, 10),
                'f5': np.random.normal(0.6, 0.12, 10),
                'f6': np.random.normal(0.4, 0.08, 10),
            }
            for sc in mi_subclasses:
                data[sc] = [1 if sc == lbl else 0 for lbl in data['label']]
            df = pd.DataFrame(data)
        else:
            df = pd.read_csv(file)
        
        # Simulasi fitur - pada implementasi nyata gunakan kolom yang sesuai
        feature_cols = [f'f{i+1}' for i in range(6)]  # Contoh kolom fitur
        
        # Pastikan kolom fitur ada
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            flash(f'Kolom fitur tidak ditemukan: {", ".join(missing_cols)}', 'danger')
            return redirect(url_for('home'))
        
        X = df[feature_cols].values
        
        # Prediksi multi-label
        # y_pred = model.predict(X)  # Pada implementasi nyata
        y_pred = predict_simulation(X)  # Untuk simulasi
        
        labels = convert_to_labels(y_pred)
        
        # Gabungkan dengan data asli
        df_result = df.copy()
        df_result["Predicted_Label"] = labels
        
        # Hitung jumlah prediksi tiap kelas
        class_counts = df_result["Predicted_Label"].value_counts().to_dict()
        
        # Hitung statistik
        total_predictions = len(df_result)
        mi_count = sum(1 for label in labels if label != 'NORM')
        norm_percentage = (total_predictions - mi_count) / total_predictions * 100
        
        return render_template(
            "results.html",
            tables=[df_result.to_html(classes='table table-striped', index=False)],
            counts=class_counts,
            total=total_predictions,
            mi_count=mi_count,
            norm_percentage=f"{norm_percentage:.1f}%",
            mi_subclasses=mi_subclasses
        )
    
    except Exception as e:
        flash(f'Error saat memproses: {str(e)}', 'danger')
        return redirect(url_for('home'))

@app.route('/sample')
def download_sample():
    # Generate sample CSV data
    data = {
        'beat_id': range(1, 11),
        'patient_id': ['P001']*10,
        'record_id': ['R001']*10,
        'label': ['NORM', 'IMI', 'AMI', 'NORM', 'LMI', 'ASMI', 'NORM', 'ILMI', 'ALMI', 'NORM'],
        'is_mi': [0,1,1,0,1,1,0,1,1,0],
        'f1': np.random.normal(0.5, 0.1, 10).round(3),
        'f2': np.random.normal(0.3, 0.05, 10).round(3),
        'f3': np.random.normal(0.7, 0.15, 10).round(3),
        'f4': np.random.normal(0.2, 0.03, 10).round(3),
        'f5': np.random.normal(0.6, 0.12, 10).round(3),
        'f6': np.random.normal(0.4, 0.08, 10).round(3),
    }
    for sc in mi_subclasses:
        data[sc] = [1 if sc == lbl else 0 for lbl in data['label']]
    
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    
    return (
        csv,
        200,
        {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=sample_data.csv'
        }
    )

if __name__ == '__main__':
    app.run(debug=True)