import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load model dan objek preprocessing
model = joblib.load('model.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Nilai rata-rata yang digunakan untuk kolom yang tidak diinputkan
mean_kelembaban_min = 65  # ganti dengan nilai yang sesuai dari data latih
mean_kelembaban_max = 95  # ganti dengan nilai yang sesuai dari data latih
mean_suhu_max = 31        # ganti dengan nilai yang sesuai dari data latih

def predict_weather(waktu, wilayah, suhu_min):
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

    # Estimasi risiko berdasarkan prediksi (misalnya, ini hanya contoh dan harus disesuaikan)
    risiko = {'Hujan': 50, 'Panas': 60, 'Berawan': 40}  # Ganti dengan logika risiko yang sesuai

    return predicted_weather_label, risiko

# Membuat aplikasi Streamlit
st.title('Prediksi Cuaca Wilayah Jakarta')

waktu_input = st.selectbox('Pilih Waktu:', ['Pagi', 'Siang', 'Malam', 'Dini Hari'])
wilayah_input = st.selectbox('Pilih Wilayah:', ['Jakarta Selatan', 'Jakarta Barat', 'Jakarta Utara', 'Jakarta Timur', 'Jakarta Pusat', 'Kepulauan Seribu'])
suhu_min_input = st.number_input('Masukkan Suhu Min:', min_value=0.0, max_value=50.0, step=0.1)

if st.button('Submit'):
    prediksi_cuaca, risiko = predict_weather(waktu_input, wilayah_input, suhu_min_input)
    st.write(f'Prediksi Cuaca: {prediksi_cuaca}')
    
    # Visualisasi risiko cuaca
    st.subheader('Grafik Risiko Cuaca')
    
    # Menyiapkan data untuk grafik
    risiko_df = pd.DataFrame(list(risiko.items()), columns=['Cuaca', 'Risiko (%)'])

    # Membuat grafik
    fig, ax = plt.subplots()
    sns.barplot(x='Cuaca', y='Risiko (%)', data=risiko_df, ax=ax)
    ax.set_title('Tingkat Risiko Cuaca')
    ax.set_ylim(0, 100)
    
    st.pyplot(fig)
