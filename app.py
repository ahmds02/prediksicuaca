from flask import Flask, request, render_template_string
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

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_weather_label = None

    if request.method == 'POST':
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

    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Prediksi Cuaca</title>
        </head>
        <body>
            <h1>Prediksi Cuaca Wilayah Jakarta</h1>
            <form method="POST">
                <label for="waktu">Waktu:</label>
                <select id="waktu" name="waktu">
                    <option value="Pagi">Pagi</option>
                    <option value="Siang">Siang</option>
                    <option value="Malam">Malam</option>
                    <option value="Dini Hari">Dini Hari</option>
                </select><br><br>

                <label for="wilayah">Wilayah:</label>
                <select id="wilayah" name="wilayah">
                    <option value="Jakarta Selatan">Jakarta Selatan</option>
                    <option value="Jakarta Barat">Jakarta Barat</option>
                    <option value="Jakarta Utara">Jakarta Utara</option>
                    <option value="Jakarta Timur">Jakarta Timur</option>
                    <option value="Jakarta Pusat">Jakarta Pusat</option>
                    <option value="Kepulauan Seribu">Kepulauan Seribu</option>
                </select><br><br>

                <label for="suhu_min">Suhu Min:</label>
                <input type="number" id="suhu_min" name="suhu_min" step="0.1" required><br><br>

                <input type="submit" value="Submit">
            </form>
            
            {% if predicted_weather_label %}
                <h2>Prediksi Cuaca: {{ predicted_weather_label }}</h2>
            {% endif %}
        </body>
        </html>
    ''', predicted_weather_label=predicted_weather_label)

if __name__ == '__main__':
    app.run(debug=True)
