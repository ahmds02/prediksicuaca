import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the pre-trained model and preprocessing objects
linear_reg = joblib.load('model.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Define a mapping for user inputs
waktu_map = {'Pagi': 0, 'Siang': 1, 'Malam': 2, 'Dini Hari': 3}
wilayah_map = {'Jakarta Selatan': 0, 'Jakarta Barat': 1, 'Jakarta Utara': 2, 'Jakarta Timur': 3, 'Jakarta Pusat': 4, 'Kepulauan Seribu': 5}

# Define the Streamlit app
st.title('Prediksi Cuaca Wilayah Jakarta')

# User inputs
waktu = st.selectbox('Waktu', ['Pagi', 'Siang', 'Malam', 'Dini Hari'])
wilayah = st.selectbox('Wilayah', ['Jakarta Selatan', 'Jakarta Barat', 'Jakarta Utara', 'Jakarta Timur', 'Jakarta Pusat', 'Kepulauan Seribu'])
suhu_min = st.number_input('Suhu Min (°C)', min_value=-50.0, step=0.1)
suhu_max = st.number_input('Suhu Max (°C)', min_value=-50.0, step=0.1)
kelembaban_min = st.number_input('Kelembaban Min (%)', min_value=0.0, step=0.1)
kelembaban_max = st.number_input('Kelembaban Max (%)', min_value=0.0, step=0.1)

# Process input data
waktu_num = waktu_map[waktu]
wilayah_num = wilayah_map[wilayah]

new_data = pd.DataFrame({
    'waktu': [waktu_num],
    'wilayah': [wilayah_num],
    'kelembaban_min': [kelembaban_min],
    'kelembaban_max': [kelembaban_max],
    'suhu_min': [suhu_min],
    'suhu_max': [suhu_max]
})

# Impute missing values and scale the features
new_data_imputed = pd.DataFrame(imputer.transform(new_data), columns=new_data.columns)
new_data_scaled = pd.DataFrame(scaler.transform(new_data_imputed), columns=new_data_imputed.columns)

# Predict weather
predicted_weather = linear_reg.predict(new_data_scaled)
predicted_weather_label = le.inverse_transform([int(round(predicted_weather[0]))])

# Display prediction result
st.subheader('Hasil Prediksi Cuaca:')
st.write(f'Cuaca yang diprediksi: {predicted_weather_label[0]}')

# Visualization of risk percentages (dummy example)
if st.checkbox('Tampilkan Grafik Risiko Cuaca'):
    # Example dummy data for visualization
    weather_risks = {'Hujan Ringan': 25, 'Hujan Sedang': 15, 'Hujan Lebat': 10, 'Cerah': 30, 'Berawan': 20}
    fig, ax = plt.subplots()
    ax.pie(weather_risks.values(), labels=weather_risks.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)