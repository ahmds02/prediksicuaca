from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load model dan objek preprocessing
model = joblib.load('model.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Nilai rata-rata yang digunakan untuk kolom yang tidak diinputkan
mean_kelembaban_min = 65  # ganti dengan nilai yang sesuai dari data latih
mean_kelembaban_max = 95  # ganti dengan nilai yang sesuai dari data latih
mean_suhu_max = 31        # ganti dengan nilai yang sesuai dari data latih

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    waktu = request.form['waktu']
    wilayah = request.form['wilayah']
    suhu_min = float(request.form['suhu_min'])

    waktu_map = {'Pagi': 0, 'Siang': 1, 'Malam': 2, 'Dini Hari': 3}
    wilayah_map = {'Jakarta Selatan': 0, 'Jakarta Barat': 1, 'Jakarta Utara': 2, 'Jakarta Timur': 3, 'Jakarta Pusat': 4, 'Kepulauan Seribu': 5}

    waktu = waktu_map[waktu]
    wilayah = wilayah_map[wilayah]

    new_data = pd.DataFrame({
        'waktu': [waktu],
        'wilayah': [wilayah],
        'kelembaban_min': [mean_kelembaban_min],
        'kelembaban_max': [mean_kelembaban_max],
        'suhu_min': [suhu_min],
        'suhu_max': [mean_suhu_max]
    })

    # Mengisi nilai NaN pada data baru
    new_data_imputed = pd.DataFrame(imputer.transform(new_data), columns=new_data.columns)

    # Melakukan penskalaan fitur pada data baru
    new_data_scaled = pd.DataFrame(scaler.transform(new_data_imputed), columns=new_data_imputed.columns)

    # Membuat prediksi cuaca berdasarkan regresi
    predicted_weather = model.predict(new_data_scaled)
    predicted_weather_label = le.inverse_transform([int(round(predicted_weather[0]))])[0]

    return f'<h1>Prediksi Cuaca: {predicted_weather_label}</h1>'

if __name__ == '__main__':
    app.run(debug=True)
