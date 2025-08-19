from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
import pickle
import os
import joblib
app = Flask(__name__)

# Simulasi model (pada implementasi nyata, gunakan model yang sebenarnya)
MODEL_PATH = "best_model_QRS+T.pkl"
model = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    mi_subclasses = ['AMI', 'IMI', 'ILMI', 'ALMI', 'ASMI', 'INFERIOR']  # Sesuaikan dengan kelasmu
    return render_template('index.html', mi_subclasses=mi_subclasses)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return redirect('/')

    df = pd.read_csv(file)
    
    # Pastikan input sesuai jumlah fitur model
    predictions = model.predict(df)

    df['Prediction'] = predictions
    df['Prediction_Label'] = df['Prediction'].apply(lambda x: 'NORM' if x == 'NORM' else str(x))

    result_path = 'static/result.csv'
    df.to_csv(result_path, index=False)

    return render_template('result.html', tables=[df.to_html(classes='table table-bordered', index=False)])

@app.route('/sample')
def sample():
    sample_path = 'static/sample.csv'
    return send_file(sample_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
