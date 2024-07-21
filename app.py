import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Load model and other necessary files
linear_reg = joblib.load('model.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Define the user input form
st.title('Prediksi Cuaca Wilayah Jakarta')

waktu = st.selectbox('Waktu', ['Pagi', 'Siang', 'Malam', 'Dini Hari'])
wilayah = st.selectbox('Wilayah', ['Jakarta Selatan', 'Jakarta Barat', 'Jakarta Utara', 'Jakarta Timur', 'Jakarta Pusat', 'Kepulauan Seribu'])
suhu_min = st.number_input('Suhu Min (°C)', min_value=0.0, step=0.1)

kelembaban_min = st.number_input('Kelembaban Min (%)', min_value=0.0, step=0.1)
kelembaban_max = st.number_input('Kelembaban Max (%)', min_value=0.0, step=0.1)
suhu_max = st.number_input('Suhu Max (°C)', min_value=0.0, step=0.1)

# Pemrosesan input
waktu_map = {'Pagi': 0, 'Siang': 1, 'Malam': 2, 'Dini Hari': 3}
wilayah_map = {'Jakarta Selatan': 0, 'Jakarta Barat': 1, 'Jakarta Utara': 2, 'Jakarta Timur': 3, 'Jakarta Pusat': 4, 'Kepulauan Seribu': 5}

waktu = waktu_map[waktu]
wilayah = wilayah_map[wilayah]

new_data = pd.DataFrame({
    'waktu': [waktu],
    'wilayah': [wilayah],
    'kelembaban_min': [kelembaban_min],
    'kelembaban_max': [kelembaban_max],
    'suhu_min': [suhu_min],
    'suhu_max': [suhu_max]
})

# Mengisi nilai NaN pada data baru
new_data_imputed = pd.DataFrame(imputer.transform(new_data), columns=new_data.columns)

# Melakukan penskalaan fitur pada data baru
new_data_scaled = pd.DataFrame(scaler.transform(new_data_imputed), columns=new_data_imputed.columns)

# Membuat prediksi cuaca berdasarkan regresi
predicted_weather = linear_reg.predict(new_data_scaled)
predicted_weather_label = le.inverse_transform([int(round(predicted_weather[0]))])

if st.button('Prediksi Cuaca'):
    st.write(f'Prediksi Cuaca: {predicted_weather_label[0]}')

# Visualisasi tambahan (opsional)
if st.checkbox('Tampilkan Grafik Distribusi Cuaca'):
    # Dummy data untuk contoh visualisasi
    weather_counts = pd.Series([5, 15, 25, 30, 10], index=['Cerah', 'Berawan', 'Hujan Ringan', 'Hujan Sedang', 'Hujan Lebat'])
    fig, ax = plt.subplots()
    weather_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    ax.set_title('Distribusi Prediksi Cuaca')
    st.pyplot(fig)
