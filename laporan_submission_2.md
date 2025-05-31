# Laporan Proyek Machine Learning - Muhammad Fadhil Diandra

## Project Overview

Dalam era digital modern, pengguna platform streaming seperti Netflix, Disney+, dan Amazon Prime dihadapkan pada ribuan pilihan film dan serial, sehingga dibutuhkan sistem yang dapat membantu menyaring konten berdasarkan preferensi pengguna. Sistem rekomendasi hadir sebagai solusi penting untuk permasalahan ini. Dengan mempelajari pola rating pengguna terhadap film, sistem ini mampu memprediksi film apa yang kemungkinan besar akan disukai pengguna berikutnya. Salah satu pendekatan yang umum digunakan adalah collaborative filtering, di mana preferensi pengguna dipelajari dari interaksi pengguna lain yang memiliki selera serupa. Metode ini telah terbukti meningkatkan pengalaman pengguna dan efisiensi layanan digital [1].

Proyek ini memanfaatkan data dari MovieLens yang terdiri atas informasi rating film dari pengguna untuk membangun model prediksi rating menggunakan pendekatan machine learning berbasis regresi. Algoritma Singular Value Decomposition (SVD) dipilih karena kemampuannya dalam melakukan dekomposisi matriks rating yang bersifat sparse, serta akurasi yang tinggi dalam memprediksi preferensi pengguna [2]. Dengan model ini, sistem dapat memperkirakan rating yang belum diberikan oleh pengguna terhadap suatu film, yang kemudian dapat digunakan untuk membuat rekomendasi yang lebih personal dan relevan. Studi oleh Netflix menunjukkan bahwa lebih dari 80% aktivitas menonton berasal dari sistem rekomendasi otomatis mereka [3], menunjukkan pentingnya akurasi dalam sistem prediksi seperti ini.z

##Referensi:
[1] F. Ricci, L. Rokach, and B. Shapira, Recommender Systems Handbook. Springer, 2015.
[2] Y. Koren, R. Bell, and C. Volinsky, “Matrix Factorization Techniques for Recommender Systems,” Computer, vol. 42, no. 8, pp. 30–37, 2009.
[3] C. A. Gomez-Uribe and N. Hunt, “The Netflix Recommender System: Algorithms, Business Value, and Innovation,” ACM Trans. Manage. Inf. Syst., vol. 6, no. 4, pp. 1–19, Dec. 2016.

## Business Understanding

### Problem Statements

- Bagaimana cara memprediksi rating yang akan diberikan pengguna terhadap film yang belum ditonton berdasarkan data historis rating yang tersedia?
- Bagaimana meningkatkan kualitas sistem rekomendasi agar pengguna mendapatkan saran film yang sesuai dengan preferensinya, sehingga meningkatkan kepuasan dan retensi pengguna pada platform?2
- Bagaimana memanfaatkan data rating yang tersedia untuk memahami preferensi pengguna terhadap genre atau jenis film tertentu?

### Goals

- Membangun model machine learning berbasis regresi untuk memprediksi rating pengguna terhadap film yang belum ditonton.
- Menghasilkan sistem rekomendasi berbasis rating prediktif agar pengguna mendapatkan rekomendasi film yang akurat dan personal.
- Menganalisis pola rating untuk mendapatkan insight terhadap genre atau tipe film yang populer atau kurang diminati oleh kelompok pengguna tertentu.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.
 
### Solution statements
- Content-Based Filtering dengan Regressor
Pendekatan ini memanfaatkan atribut film (seperti genre, tahun rilis, dll) dan preferensi pengguna berdasarkan histori film yang pernah ditonton untuk membangun model prediksi rating menggunakan algoritma regresi seperti Linear Regression atau Random Forest Regressor. Namun karena dalam dataset ini hanya tersedia data rating dan tidak lengkap untuk metadata film, pendekatan ini tidak digunakan pada implementasi utama.

- Collaborative Filtering dengan Matrix Factorization (SVD)
Metode ini menggunakan pendekatan kolaboratif dengan menganalisis pola rating antar pengguna untuk menyarankan film yang mungkin disukai oleh pengguna lain dengan pola serupa. Algoritma seperti Singular Value Decomposition (SVD) menjadi solusi utama dalam proyek ini karena mampu menangkap hubungan laten antara pengguna dan film berdasarkan rating historis.

- Baseline Predictive Model
Sebagai pembanding, digunakan juga pendekatan sederhana seperti memprediksi rata-rata rating film atau rata-rata rating pengguna untuk menilai seberapa jauh peningkatan performa dari model utama.

## Data Understanding
Sumber data : https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/data
Dataset yang digunakan dalam proyek ini berasal dari MovieLens, tepatnya versi MovieLens 100k Dataset, yang berisi data rating film dari 943 pengguna terhadap 1682 film. Dataset ini cocok digunakan untuk membangun sistem rekomendasi dan tugas prediksi rating karena berisi lebih dari 100.000 entri rating yang bersifat kuantitatif. Dalam proyek ini, hanya dua file utama yang digunakan, yaitu:
- movies.csv — berisi metadata film seperti ID film, judul, dan genre.
- ratings.csv — berisi data rating dari pengguna terhadap film dengan timestamp.

Total data dalam ratings.csv adalah 100.836 entri rating, sedangkan movies.csv mencakup 9742 film, namun hanya sebagian yang memiliki rating (karena keterbatasan interaksi pengguna). Data ini telah memenuhi syarat minimum kuantitatif (>500 sampel) dan sangat relevan untuk pendekatan regresi dan sistem rekomendasi.

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
- Mayoritas rating berada di angka 3.0 hingga 4.0, dengan puncak di rating 4.0.
- Ini menunjukkan adanya kecenderungan pengguna memberi nilai cukup tinggi pada film yang mereka pilih untuk ditonton.
2. Jumlah Rating per Film
- Terdapat ketimpangan jumlah rating: beberapa film populer menerima ratusan rating, sedangkan sebagian besar film hanya menerima sedikit rating.
- Hal ini mengindikasikan pentingnya menangani sparsity data saat membangun model rekomendasi.
3. Jumlah Rating per Pengguna
- Sebagian besar pengguna memberikan antara 20 hingga 200 rating.
- Variasi ini penting diperhatikan karena dapat mempengaruhi generalisasi model prediktif terhadap pengguna baru (cold-start problem).
4. Genre Paling Umum
- Genre seperti Drama, Comedy, dan Action termasuk yang paling sering muncul dalam dataset, menunjukkan preferensi umum pengguna.
- Visualisasi seperti histogram distribusi rating, bar chart genre, dan heatmap user-movie matrix digunakan untuk membantu memahami tren dan struktur data secara lebih intuitif.

Visualisasi seperti histogram distribusi rating, bar chart genre, dan heatmap user-movie matrix digunakan untuk membantu memahami tren dan struktur data secara lebih intuitif.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

1. Merge Data
Data film dan data rating pengguna digabungkan (merge) berdasarkan movieId agar setiap baris data memiliki informasi lengkap tentang judul film, serta rating yang diberikan pengguna. Ini penting untuk menggabungkan informasi fitur dan label dalam satu frame data.
2. Handling Missing Values
Dilakukan pengecekan dan penanganan nilai hilang (missing values) untuk mencegah error atau bias saat proses pelatihan model. Jika ada nilai kosong, biasanya dihapus atau diisi dengan nilai tertentu tergantung konteks dan jumlah yang hilang.
3. Feature Engineering
Dilakukan pembuatan fitur tambahan seperti tahun rilis film yang diekstrak dari judul. Fitur-fitur seperti ini dapat membantu model mengenali pola perilaku pengguna terhadap film berdasarkan waktu atau genre tertentu.
4. Encoding Fitur Kategorikal
Fitur-fitur kategorikal seperti genre atau userId diubah menjadi representasi numerik agar bisa digunakan oleh algoritma machine learning. Contohnya, userId dan movieId di-encode dengan Label Encoding untuk digunakan dalam model matrix factorization.
5. Feature 
Hanya fitur-fitur yang relevan seperti userId, movieId, dan rating yang dipertahankan untuk modeling. Fitur lain yang tidak berkontribusi terhadap prediksi rating disaring agar model menjadi lebih efisien.
6. Train-Test Split
Dataset dibagi menjadi data latih dan data uji untuk mengevaluasi performa model secara adil. Biasanya dilakukan dengan proporsi 80:20.

## Modeling
Pada tahap ini, dilakukan penerapan dua pendekatan sistem rekomendasi untuk memprediksi rating yang diberikan pengguna terhadap film, dan menghasilkan Top-N Recommendation.

Dua pendekatan utama yang digunakan adalah:

1. Content-Based Filtering (CBF)
Pendekatan ini merekomendasikan film kepada pengguna berdasarkan kemiripan konten film yang telah mereka sukai sebelumnya.
- Implementasi:
    - Menggunakan kolom genres dari data movies.csv.
    - Dilakukan transformasi teks genre menjadi representasi vektor menggunakan TF-IDF.
    - Hitung kemiripan antar film menggunakan cosine similarity.
    - Diberikan rekomendasi film serupa berdasarkan genre film yang telah ditonton pengguna.

2. Matrix Factorization - SVD
Pendekatan collaborative filtering ini menggunakan interaksi pengguna terhadap film dalam bentuk matriks, lalu dilakukan dekomposisi matriks untuk menemukan pola laten antara pengguna dan item.

- Implementasi:
Menggunakan library Surprise untuk menerapkan algoritma SVD:


## Evaluation
Dalam mengevaluasi performa sistem rekomendasi yang telah dibangun, digunakan dua metrik utama yang umum dipakai pada sistem rekomendasi prediktif, yaitu:
- Root Mean Squared Error (RMSE)
RMSE 0.87 menandakan bahwa rata-rata kesalahan prediksi model sekitar 0.87 poin pada skala - rating 0.5–5.0.
- Mean Absolute Error (MAE)
MAE 0.67 menunjukkan bahwa secara rata-rata, prediksi menyimpang 0.67 dari nilai rating aktual pengguna.

Hasil ini menunjukkan bahwa model cukup baik dalam memprediksi rating film yang akan diberikan pengguna, dengan kesalahan di bawah 1 poin. Metrik RMSE dan MAE juga konsisten menunjukkan bahwa model SVD dapat menangkap preferensi pengguna dengan cukup akurat.

Sebagai perbandingan, pendekatan Content-Based Filtering tidak menghasilkan prediksi rating secara numerik, sehingga evaluasinya lebih kualitatif (berdasarkan relevansi subyektif hasil rekomendasi). Namun demikian, sistem ini tetap bermanfaat terutama dalam kondisi cold-start.


