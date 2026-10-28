- **Nama:** Nabil Pratama Alfarisi
- **Nim:** 241011402593
- **Kelas:** 05TPLP023
- **Prodi:** Teknik Informatika




# Prediksi Harga Rumah dengan Machine Learning

## 1. Deskripsi Proyek

Proyek ini bertujuan untuk membuat model Machine Learning yang dapat memperkirakan harga rumah berdasarkan karakteristik rumah, seperti kualitas rumah, kondisi bangunan, tahun pembangunan, luas area tertentu, dan fitur lainnya.

Secara sederhana, model akan belajar dari data rumah yang sudah memiliki harga jual. Setelah mempelajari pola tersebut, model digunakan untuk memperkirakan harga rumah dari data yang belum diketahui harganya.

Proyek ini mencakup tiga tahapan utama:

1. **Modeling** — membuat dan melatih model Machine Learning.
2. **Evaluation** — mengukur dan membandingkan performa model.
3. **Deployment** — menggunakan model melalui aplikasi sederhana dengan Streamlit.

## 2. Dataset yang Digunakan

Dataset dalam proyek ini terdiri dari beberapa file:

- `train.csv`: data rumah yang memiliki harga jual dan digunakan untuk melatih model.
- `test.csv`: data rumah yang digunakan untuk menghasilkan prediksi.
- `data_description.txt`: penjelasan mengenai arti kolom atau fitur yang ada dalam dataset.
- `sample_submission.csv`: contoh format file hasil prediksi untuk kebutuhan pengumpulan data.

Kolom `SalePrice` merupakan target yang ingin diprediksi, sedangkan kolom lainnya berisi informasi tentang karakteristik rumah.

Kolom `Id` hanya digunakan sebagai identitas setiap rumah, sehingga tidak digunakan sebagai fitur untuk memprediksi harga.

## 3. Tahap Pertama: Modeling

### Apa itu modeling?

Modeling adalah proses membuat dan melatih model Machine Learning agar komputer dapat mempelajari hubungan antara karakteristik rumah dan harga jualnya.

Sebagai contoh, model dapat mempelajari pola hubungan antara kualitas rumah, kondisi bangunan, dan tahun pembangunan dengan harga rumah yang tercatat dalam dataset.

### Langkah-langkah modeling

**Langkah 1 — Memuat dataset**

Data `train.csv` dibaca untuk memperoleh karakteristik rumah dan harga jual sebenarnya. Data `test.csv` juga dibaca untuk menyiapkan rumah yang akan diprediksi.

**Langkah 2 — Memisahkan fitur dan target**

Data dibagi menjadi dua bagian:

- **Fitur (X):** informasi mengenai karakteristik rumah yang digunakan untuk membuat prediksi.
- **Target (y):** harga jual rumah sebenarnya, yaitu `SalePrice`.

Kolom `SalePrice` dan `Id` tidak dimasukkan ke dalam fitur `X`. `SalePrice` merupakan nilai yang ingin diprediksi, sedangkan `Id` hanya berfungsi sebagai identitas data.

**Langkah 3 — Membagi data pelatihan dan pengujian**

Data yang memiliki harga jual dibagi menjadi data pelatihan (*training data*) dan data pengujian (*testing data*).

- Data pelatihan digunakan untuk mengajari model.
- Data pengujian digunakan untuk mengukur kemampuan model memprediksi data yang tidak digunakan selama pelatihan.

Pembagian ini penting agar performa model tidak hanya dinilai dari data yang sudah dipelajarinya.

**Langkah 4 — Melatih model**

Dalam proyek ini, dua algoritma Machine Learning digunakan untuk dibandingkan.

1. **Linear Regression**

   Linear Regression memperkirakan harga berdasarkan hubungan matematis antara karakteristik rumah dan harga jual. Model ini dapat menjadi pendekatan awal yang sederhana dan mudah dijelaskan.

2. **Random Forest Regressor**

   Random Forest menggunakan gabungan banyak pohon keputusan untuk menghasilkan prediksi. Model ini dapat mempelajari pola hubungan yang lebih kompleks dalam data.

Kedua model dilatih menggunakan data pelatihan yang sama agar performanya dapat dibandingkan.

**Langkah 5 — Menghasilkan prediksi**

Setelah dilatih, kedua model digunakan untuk memprediksi harga rumah pada data pengujian. Hasil prediksi kemudian dibandingkan dengan harga sebenarnya menggunakan metrik evaluasi.

## 4. Tahap Kedua: Evaluation

### Apa itu evaluasi model?

Evaluasi adalah proses mengukur seberapa baik model memprediksi harga rumah.

Misalnya, jika harga sebenarnya sebuah rumah adalah $200.000, sedangkan model memprediksi $190.000, berarti terdapat selisih sebesar $10.000.

Untuk menilai performa model secara lebih menyeluruh, proyek ini menggunakan tiga metrik, yaitu MAE, RMSE, dan R².

### A. MAE (Mean Absolute Error)

MAE mengukur rata-rata selisih absolut antara harga prediksi dan harga sebenarnya.

Dengan bahasa sederhana, MAE menjawab pertanyaan:

**“Rata-rata, seberapa jauh prediksi model dari harga rumah sebenarnya?”**

Contohnya, jika MAE sebesar $10.000, rata-rata selisih absolut prediksi dengan harga sebenarnya adalah sekitar $10.000.

- Semakin kecil MAE, semakin baik.
- MAE mudah dipahami karena menggunakan satuan yang sama dengan harga rumah.
- MAE tidak memberikan penalti sebesar RMSE terhadap kesalahan yang sangat besar.

### B. RMSE (Root Mean Squared Error)

RMSE mengukur kesalahan prediksi dengan memberikan pengaruh lebih besar pada kesalahan yang besar.

Dengan bahasa sederhana, RMSE menjawab pertanyaan:

**“Seberapa besar kesalahan prediksi jika kesalahan yang sangat besar ingin lebih diperhatikan?”**

Misalnya, kesalahan prediksi sebesar $50.000 akan memberikan pengaruh yang lebih besar pada RMSE daripada kesalahan sebesar $5.000.

- Semakin kecil RMSE, semakin baik.
- RMSE menggunakan satuan yang sama dengan harga rumah.
- RMSE cocok ketika kesalahan prediksi yang sangat besar perlu mendapat perhatian lebih.

### C. R² (R-squared)

R² mengukur seberapa baik model menjelaskan variasi harga rumah dalam data evaluasi.

Dengan bahasa sederhana, R² menjawab pertanyaan:

**“Seberapa baik model menangkap pola perbedaan harga rumah dalam data?”**

Secara umum:

- R² mendekati 1 menunjukkan kemampuan yang lebih baik dalam menjelaskan variasi harga.
- R² mendekati 0 menunjukkan bahwa model tidak banyak menjelaskan variasi harga dibandingkan pendekatan sederhana menggunakan rata-rata.
- R² negatif berarti performa model dapat lebih buruk daripada pendekatan rata-rata tersebut pada data evaluasi.

R² bukan persentase prediksi yang pasti benar. Nilai R² sebesar 0,89, misalnya, tidak berarti 89% prediksi model pasti akurat.

### Perbandingan MAE, RMSE, dan R²

| Metrik | Fungsi | Hasil yang diharapkan |
|---|---|---|
| MAE | Mengukur rata-rata besar kesalahan prediksi | Semakin kecil |
| RMSE | Mengukur kesalahan dan lebih memperhatikan kesalahan besar | Semakin kecil |
| R² | Mengukur kemampuan model menjelaskan variasi harga | Semakin tinggi |

### Metrik mana yang paling cocok untuk prediksi harga rumah?

Dalam proyek ini, **RMSE digunakan sebagai metrik utama untuk memilih model terbaik**, sedangkan MAE dan R² tetap digunakan sebagai metrik pendukung.

Alasannya, kesalahan prediksi harga rumah yang sangat besar dapat menjadi masalah penting. RMSE memberikan penalti lebih besar terhadap kesalahan semacam itu sehingga membantu membedakan model yang sesekali membuat prediksi sangat meleset.

Namun, tidak ada satu metrik yang selalu paling baik untuk semua kebutuhan. MAE tetap penting karena lebih mudah menjelaskan rata-rata besar kesalahan, sedangkan R² membantu memahami seberapa baik model menangkap variasi harga.

Karena itu, ketiga metrik digunakan bersama agar evaluasi model lebih lengkap.

### Hasil Evaluasi Model

Berdasarkan hasil yang ditampilkan pada notebook, model yang terpilih adalah **Random Forest**, karena memiliki nilai RMSE terendah di antara model yang dibandingkan.

Hasil evaluasi Random Forest adalah sebagai berikut:

| Metrik | Hasil |
|---|---:|
| MAE | 17.335,894726 |
| RMSE | 28.443,450577 |
| R² | 0,894525 |

**Interpretasi hasil:**

- **MAE sebesar 17.335,89:** rata-rata selisih absolut antara prediksi dan harga sebenarnya sekitar 17.336 satuan harga.
- **RMSE sebesar 28.443,45:** menunjukkan ukuran kesalahan prediksi dengan penalti lebih besar untuk kesalahan yang besar. Nilai RMSE yang lebih tinggi daripada MAE dapat mengindikasikan adanya sebagian kesalahan prediksi yang cukup besar.
- **R² sebesar 0,894525:** menunjukkan bahwa model menjelaskan sekitar 89,45% variasi harga rumah pada data evaluasi.

Angka MAE dan RMSE mengikuti satuan harga yang digunakan dalam dataset. Pada aplikasi, harga ditampilkan dalam dolar.

Berdasarkan hasil tersebut, Random Forest dipilih untuk digunakan pada tahap selanjutnya. Hasil evaluasi ini menggambarkan performa pada data pengujian yang digunakan, bukan jaminan bahwa setiap prediksi rumah baru akan memiliki kesalahan yang sama.

## 5. Tahap Ketiga: Menyimpan Model Terbaik

Setelah evaluasi dilakukan, model yang dipilih dilatih kembali menggunakan seluruh data pelatihan yang memiliki harga jual.

Tujuannya agar model dapat memanfaatkan seluruh data berlabel yang tersedia sebelum digunakan untuk menghasilkan prediksi.

Model kemudian disimpan dalam file:

`house_price_model.joblib`

File ini berisi model yang telah dilatih sehingga aplikasi tidak perlu mengulang proses pelatihan setiap kali dijalankan.

Model yang disimpan kemudian digunakan untuk menghasilkan prediksi harga rumah pada `test.csv`. Hasil prediksi juga disimpan dalam file:

`house_price_predictions.csv`

## 6. Tahap Keempat: Deployment Menggunakan Streamlit

### Apa itu deployment?

Deployment adalah tahap ketika model Machine Learning dibuat agar bisa digunakan melalui sebuah aplikasi, bukan hanya dijalankan di notebook.

Dalam proyek ini, Streamlit digunakan untuk membuat aplikasi web sederhana yang menampilkan karakteristik rumah dan hasil estimasi harganya.

### Cara kerja aplikasi

Aplikasi `app.py` melakukan langkah-langkah berikut:

1. Memuat model dari `house_price_model.joblib`.
2. Membaca data rumah dari `test.csv`.
3. Menyediakan pilihan nomor baris rumah yang ingin diprediksi.
4. Menampilkan karakteristik rumah yang dipilih.
5. Menjalankan prediksi ketika tombol **Prediksi harga** ditekan.
6. Menampilkan estimasi harga rumah dalam bentuk angka dolar.

Aplikasi ini masih berupa demo sederhana. Pengguna memilih rumah dari data yang sudah tersedia dalam `test.csv`; aplikasi belum menyediakan formulir untuk memasukkan karakteristik rumah secara bebas.

### Langkah-langkah menjalankan aplikasi

**Langkah 1 — Menyiapkan file**

Pastikan file berikut tersedia dalam satu folder:

- `app.py`
- `house_price_model.joblib`
- `test.csv`

**Langkah 2 — Membuka terminal**

Buka terminal pada folder proyek yang berisi file aplikasi.

**Langkah 3 — Menginstal library**

Jalankan perintah berikut jika library yang diperlukan belum terpasang:

```bash
python -m pip install streamlit pandas scikit-learn joblib
```

**Langkah 4 — Menjalankan Streamlit**

Jalankan perintah berikut:

```bash
python -m streamlit run app.py
```

**Langkah 5 — Membuka aplikasi di browser**

Streamlit akan menjalankan aplikasi secara lokal. Biasanya, aplikasi dapat dibuka melalui alamat:

`http://localhost:8501`

**Langkah 6 — Melakukan prediksi**

Pilih nomor baris rumah yang ingin diprediksi, periksa karakteristik rumah yang ditampilkan, lalu klik tombol **Prediksi harga**.

Aplikasi akan menampilkan estimasi `SalePrice` berdasarkan model yang telah dilatih.

Sebagai contoh, pada pengujian aplikasi, model menghasilkan estimasi harga sebesar **$129,861** untuk baris data yang dipilih.

Angka tersebut merupakan hasil prediksi untuk satu rumah, bukan nilai MAE, RMSE, R², atau ukuran performa keseluruhan model. Hasil ini juga bukan penilaian resmi harga properti.

## 7. Referensi Ilmiah

Dua artikel ilmiah berikut digunakan sebagai referensi untuk mendukung pembahasan mengenai prediksi harga properti, penerapan Machine Learning, dan evaluasi model.

### Referensi 1

Mathotaarachchi, K. V., Hasan, R., & Mahmood, S. (2024). *Advanced Machine Learning Techniques for Predictive Modeling of Property Prices.* Information, 15(6), 295.

**Tautan artikel:** https://www.mdpi.com/2078-2489/15/6/295

Artikel ini membahas penerapan teknik Machine Learning untuk memprediksi harga properti. Artikel tersebut relevan sebagai landasan untuk memahami penggunaan dan perbandingan model prediksi harga rumah.

### Referensi 2

An, S., Song, Y., Jang, H., et al. (2025). *Toward transparent and accurate housing price appraisal: Hedonic price models versus machine learning algorithms.* Financial Innovation, 11, Article 141.

**Tautan artikel:** https://link.springer.com/article/10.1186/s40854-025-00874-w

Artikel ini membahas penilaian harga rumah menggunakan pendekatan tradisional dan algoritma Machine Learning, termasuk Random Forest. Artikel ini juga relevan dengan pembahasan metrik evaluasi, seperti RMSE dan R², serta penggunaan MAE sebagai metrik tambahan.

Kedua referensi tersebut mendukung landasan teori proyek. Namun, performa model dalam proyek ini tetap ditentukan berdasarkan hasil evaluasi dataset yang digunakan, bukan berdasarkan hasil penelitian dari artikel tersebut.

## 8. Kesimpulan

Proyek ini menerapkan tahapan dasar pengembangan Machine Learning untuk memprediksi harga rumah, mulai dari modeling, evaluation, hingga deployment.

Pada tahap modeling, Linear Regression dan Random Forest digunakan untuk mempelajari hubungan antara karakteristik rumah dan harga jualnya.

Pada tahap evaluation, performa kedua model dibandingkan menggunakan MAE, RMSE, dan R². RMSE dijadikan metrik utama untuk memilih model karena lebih memperhatikan kesalahan prediksi yang besar. Berdasarkan hasil evaluasi pada notebook, Random Forest terpilih dengan MAE sebesar 17.335,894726, RMSE sebesar 28.443,450577, dan R² sebesar 0,894525.

Pada tahap deployment, model disimpan dalam format `joblib` dan digunakan pada aplikasi Streamlit. Aplikasi memungkinkan pengguna memilih data rumah dari `test.csv` dan melihat estimasi harga rumah secara langsung.

Dengan demikian, proyek ini mencakup tiga tahap utama pengembangan Machine Learning: **modeling, evaluation, dan deployment**.

Pengembangan selanjutnya dapat dilakukan dengan menambahkan formulir input karakteristik rumah, memperbaiki penanganan data yang kosong, melakukan penyetelan parameter model, dan menguji model pada data yang lebih beragam.
#   L a t i h a n - M a c h i n e - L e a r n i n g - s m s 5  
 