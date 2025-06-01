# Laporan Proyek Machine Learning - Muhammad Fadhil Diandra

## Project Overview

Dalam era digital saat ini, platform streaming seperti Netflix, Disney+, dan Amazon Prime menawarkan ribuan judul film dan serial. Pengguna kerap kesulitan memilih konten yang sesuai preferensi mereka. Sistem rekomendasi hadir sebagai solusi penting untuk menyaring konten berdasarkan preferensi dan perilaku pengguna. Dengan menganalisis pola rating pengguna terhadap film, sistem ini dapat memprediksi film yang kemungkinan besar akan disukai pengguna di masa depan.

Salah satu pendekatan populer adalah collaborative filtering, yaitu mempelajari interaksi pengguna dan menemukan kesamaan perilaku dengan pengguna lain. Dalam proyek ini, pendekatan matrix factorization menggunakan algoritma Singular Value Decomposition (SVD) diterapkan untuk memprediksi rating film yang belum ditonton pengguna. SVD dipilih karena kemampuannya mengatasi matriks yang sparse dan memberikan hasil prediksi yang akurat [2].

Model ini dibangun menggunakan data dari MovieLens 100k yang menyediakan lebih dari 100.000 rating pengguna terhadap film. Output model berupa prediksi rating akan digunakan untuk memberikan rekomendasi film yang bersifat personal dan relevan bagi setiap pengguna. Studi oleh Netflix menunjukkan bahwa lebih dari 80% aktivitas menonton berasal dari sistem rekomendasi otomatis mereka [3], memperkuat pentingnya sistem seperti ini dalam meningkatkan pengalaman pengguna.

## Referensi:
- [1] F. Ricci, L. Rokach, and B. Shapira, Recommender Systems Handbook. Springer, 2015.
- [2] Y. Koren, R. Bell, and C. Volinsky, “Matrix Factorization Techniques for Recommender Systems,” Computer, vol. 42, no. 8, pp. 30–37, 2009.
- [3] C. A. Gomez-Uribe and N. Hunt, “The Netflix Recommender System: Algorithms, Business Value, and Innovation,” ACM Trans. Manage. Inf. Syst., vol. 6, no. 4, pp. 1–19, Dec. 2016.

## Business Understanding

### Problem Statements

- Bagaimana cara memprediksi rating yang akan diberikan pengguna terhadap film yang belum ditonton berdasarkan data historis rating yang tersedia?
- Bagaimana meningkatkan kualitas sistem rekomendasi agar pengguna menerima saran film yang relevan dengan preferensinya?
- Bagaimana memanfaatkan data rating untuk memahami kecenderungan pengguna terhadap genre film tertentu?

### Goals

- Membangun model machine learning berbasis regresi untuk memprediksi rating pengguna terhadap film.
- Menghasilkan sistem rekomendasi berbasis prediksi rating yang akurat dan personal.
- Menganalisis pola rating pengguna untuk menemukan insight genre atau jenis film yang populer.
 
### Solution statements
- Collaborative Filtering dengan Matrix Factorization (SVD)
Digunakan sebagai pendekatan utama. Dengan menggunakan interaksi historis antar pengguna dan film, model mampu mempelajari pola laten untuk menghasilkan rekomendasi yang akurat.

- Content-Based Filtering (CBF)
Digunakan sebagai pelengkap. Model ini merekomendasikan film yang mirip dengan yang disukai pengguna berdasarkan genre. Tidak menghasilkan prediksi numerik, tapi berguna saat cold-start.

- Baseline Predictive Model
Sebagai pembanding, digunakan model sederhana seperti rata-rata rating film atau pengguna untuk melihat seberapa besar peningkatan performa model utama.


## Data Understanding
Sumber data : https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset
- movie.csv (≈27.278 baris) berisi kolom movieId, title, dan genres.
- rating.csv (≈2.369.902 baris) berisi kolom userId, movieId, rating, dan timestamp

Hasil pemeriksaan:
- Tidak terdapat missing value di kedua file.
- movies memiliki 27.278 film, ratings memiliki 2.369.902 entri rating.

### Variabel dalam Dataset
Berikut adalah penjelasan masing-masing fitur:

File: ratings.csv
- userId : ID unik yang mewakili pengguna.
- movieId : ID unik film yang diberikan rating.
- rating : Skor rating dari pengguna terhadap film (rentang 0.5 hingga 5.0).
- timestamp : Waktu (dalam format UNIX time) saat pengguna memberikan rating.

File: movies.csv
- movieId : ID unik film (kunci relasi ke ratings.csv).
- title : Judul film (mengandung tahun rilis dalam tanda kurung).
- genres : Daftar genre yang dikaitkan dengan film, dipisahkan oleh simbol |.

### Exploratory Data Analysis (EDA):

1. Distribusi Rating
- Mayoritas rating berada di kisaran 3.0–4.0, dengan puncak di sekitar 3.5–4.0.
2. Jumlah Rating per Film
- Film‐film paling populer (dilihat dari banyaknya rating) mencakup judul‐judul klasik dan blockbuster.
3. Jumlah Rating per Pengguna
- Sebagian besar pengguna memberikan antara 20–200 rating, menunjukkan variasi aktivitas rating yang cukup besar.

Visualisasi yang digunakan:
- Histogram Distribusi Rating
- Bar Chart 10 Film dengan Jumlah Rating Terbanyak
- Histogram Jumlah Rating per Pengguna

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

Library Surprise memerlukan format khusus: tiga kolom (userId, movieId, rating) beserta objek Reader untuk mendefinisikan rentang rating.

## Modeling
Model SVD digunakan untuk mempelajari pola laten interaksi pengguna–film dari matriks rating yang sangat sparse.

model = SVD()
model.fit(trainset)
predictions = model.test(testset)


## Evaluation
Dua metrik yang dihitung: Root Mean Squared Error (RMSE) dan Mean Absolute Error (MAE).

rmse = accuracy.rmse(predictions)  
- RMSE: 0.8156
mae  = accuracy.mae(predictions)
- MAE : 0.6235

Hasil ini menunjukkan bahwa model cukup baik dalam memprediksi rating film yang akan diberikan pengguna, dengan kesalahan di bawah 1 poin. Metrik RMSE dan MAE juga konsisten menunjukkan bahwa model SVD dapat menangkap preferensi pengguna dengan cukup akurat.

Dengan menambahkan CBF dan Baseline Predictive Model, laporanmu menjadi lebih lengkap dan kuat secara metodologi. Kamu bisa juga bandingkan hasil evaluasinya (RMSE & MAE) dengan model SVD sebelumnya.

