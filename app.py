import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    model = joblib.load("model_luas_tanaman.pkl")
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("luastanaman.csv")
    return df

model = load_model()
df = load_data()


st.title("Prediksi Luas Tanaman di Jawa Barat")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning – Linear Regression** untuk memprediksi
**luas tanaman (hektar)** berdasarkan:
- Nama Kabupaten/Kota  
- Kondisi tanaman  
- Tahun  
""")

st.sidebar.header("Info Dataset")
st.sidebar.write(f"Jumlah data: **{len(df)}** baris")
st.sidebar.write("Kolom utama: `nama_kabupaten_kota`, `kondisi_tanaman`, `tahun`, `luas_tanaman`")

st.subheader("Visualisasi Data")

tab1, tab2, tab3 = st.tabs(["Distribusi Luas", "Tren per Tahun", "Rata-rata per Kabupaten"])

with tab1:
    st.write("Distribusi luas tanaman (hektar):")
    st.bar_chart(df["luas_tanaman"].value_counts().sort_index())

with tab2:
    st.write("Rata-rata luas tanaman per tahun:")
    mean_by_year = df.groupby("tahun")["luas_tanaman"].mean().reset_index()
    st.line_chart(mean_by_year, x="tahun", y="luas_tanaman")

with tab3:
    st.write("Top 10 kabupaten/kota dengan rata-rata luas tanaman terbesar:")
    mean_by_kab = (
        df.groupby("nama_kabupaten_kota")["luas_tanaman"]
        .mean()
        .reset_index()
        .sort_values("luas_tanaman", ascending=False)
        .head(10)
    )
    st.bar_chart(mean_by_kab, x="nama_kabupaten_kota", y="luas_tanaman")

with st.expander("Lihat sampel data mentah"):
    st.dataframe(df.head(20))


st.subheader("Simulasi Prediksi Luas Tanaman")

kabupaten_list = sorted(df["nama_kabupaten_kota"].unique())
kondisi_list = sorted(df["kondisi_tanaman"].unique())
tahun_min = int(df["tahun"].min())
tahun_max = int(df["tahun"].max())

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)
    with col1:
        nama_kabupaten_kota = st.selectbox("Pilih Kabupaten/Kota", kabupaten_list)
    with col2:
        kondisi_tanaman = st.selectbox("Pilih Kondisi Tanaman", kondisi_list)

    tahun = st.slider("Pilih Tahun", tahun_min, tahun_max, tahun_max)

    submit = st.form_submit_button("Prediksi Luas Tanaman")

if submit:
    input_df = pd.DataFrame({
        "nama_kabupaten_kota": [nama_kabupaten_kota],
        "kondisi_tanaman": [kondisi_tanaman],
        "tahun": [tahun]
    })

    prediksi = model.predict(input_df)[0]
    st.success(f"Perkiraan luas tanaman: **{prediksi:.2f} hektar**")

    st.caption("Ini adalah hasil prediksi dari model Linear Regression yang sudah dilatih di Google Colab.")

