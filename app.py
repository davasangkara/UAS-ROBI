import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model():
    return joblib.load("model_luas_tanaman.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("luastanaman.csv")


model = load_model()
df = load_data()

st.title("Prediksi Luas Tanaman di Jawa Barat")
st.write("Metode:Linear Regression")


st.sidebar.header("Informasi Dataset")
st.sidebar.write(f"Jumlah data: {len(df)} baris")
st.sidebar.write(
    "Kolom yang digunakan: `nama_kabupaten_kota`, `kondisi_tanaman`, `tahun` → `luas_tanaman`")


st.subheader("Input Data Prediksi")

kabupaten_list = sorted(df["nama_kabupaten_kota"].unique())
kondisi_list = sorted(df["kondisi_tanaman"].unique())
tahun_min = int(df["tahun"].min())
tahun_max = int(df["tahun"].max())

col1, col2 = st.columns(2)

with col1:
    nama_kabupaten_kota = st.selectbox("Kabupaten/Kota", kabupaten_list)

with col2:
    kondisi_tanaman = st.selectbox("Kondisi Tanaman", kondisi_list)

tahun = st.slider("Tahun", tahun_min, tahun_max, tahun_max)

if st.button("Prediksi Luas Tanaman"):

    input_df = pd.DataFrame({
        "nama_kabupaten_kota": [nama_kabupaten_kota],
        "kondisi_tanaman": [kondisi_tanaman],
        "tahun": [tahun]
    })

    prediksi = model.predict(input_df)[0]

    st.success(f"Perkiraan luas tanaman: **{prediksi:.2f} hektar**")

    st.caption(
        "Catatan: Nilai ini adalah estimasi berdasarkan pola historis data yang digunakan saat training.")

with st.expander("Lihat sampel data asli"):
    st.dataframe(df.head(20))
