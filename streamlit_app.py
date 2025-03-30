import streamlit as st
import pandas as pd
import joblib

# Modeli və column siyahısını yüklə
model = joblib.load("model.pkl")
expected_columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Avtomobil Qiyməti Proqnoz", layout="centered")
st.title("🚗 Turbo.az Avtomobil Qiyməti Proqnozlayıcı")

st.markdown("""
Bu tətbiq sizin daxil etdiyiniz avtomobil məlumatları əsasında 
təxmini qiymət həsablaır.
""")

# Input sahələri
year = st.number_input("Buraxılış ili", min_value=1990, max_value=2024, value=2015)
mileage = st.number_input("Yürüş (km)", min_value=0, max_value=1000000, value=150000)
engine = st.number_input("Mühərrik həcmi (L)", min_value=0.5, max_value=6.5, value=2.0, step=0.1)

brand = st.selectbox("Marka", ["Hyundai", "Kia", "Toyota", "Mercedes", "BMW", "LADA", "Opel", "Volkswagen"])
fuel = st.selectbox("Yanacaq növü", ["Benzin", "Dizel", "Hibrid", "Qaz"])
gearbox = st.selectbox("Sürətlər qutusu", ["Avtomat", "Mexaniki"])
city = st.selectbox("Şəhər", ["Baki", "Gəncə", "Sumqayıt", "Mingəçevir", "Şamaxi"])

# Input-u dataframeə çevir
input_df = pd.DataFrame({
    "year": [year],
    "mileage": [mileage],
    "engine": [engine],
    "brand": [brand],
    "fuel": [fuel],
    "gearbox": [gearbox],
    "city": [city]
})

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(pd.DataFrame(columns=expected_columns))
input_encoded = pd.get_dummies(input_df)

# Çatmayan sütunları 0-lamaq
for col in expected_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Sütunları düzgün qaydada tərtib et
input_encoded = input_encoded[expected_columns]

# Proqnoz
if st.button("Qiyməti Hesabla"):
    try:
        prediction = model.predict(input_encoded)
        st.success(f"Təxmini qiymət: {int(prediction[0]):,} AZN")
    except Exception as e:
        st.error(f"Xəta baş verdi: {e}")
