
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Harga Rumah", page_icon="🏠")
st.title("Prediksi Harga Rumah")
st.write("Demo sederhana model Machine Learning untuk estimasi harga rumah.")

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.joblib")

@st.cache_data
def load_test_data():
    return pd.read_csv("test.csv")

try:
    model = load_model()
    test_data = load_test_data()
    input_data = test_data.drop(columns=["Id"], errors="ignore")

    st.write("Pilih satu baris contoh dari test.csv untuk dibuat prediksi.")
    row_index = st.number_input(
        "Nomor baris (mulai dari 0)",
        min_value=0,
        max_value=max(0, len(input_data) - 1),
        value=0,
        step=1
    )
    sample = input_data.iloc[[int(row_index)]]
    st.subheader("Karakteristik rumah")
    st.dataframe(sample.T.rename(columns={sample.index[0]: "Nilai"}))

    if st.button("Prediksi harga"):
        prediction = model.predict(sample)[0]
        st.success(f"Estimasi SalePrice: ${prediction:,.0f}")
        st.caption("Ini adalah estimasi model, bukan penilaian resmi harga properti.")
except Exception as e:
    st.error("Aplikasi gagal dimuat. Pastikan app.py, house_price_model.joblib, dan test.csv berada dalam folder yang sama.")
    st.exception(e)
