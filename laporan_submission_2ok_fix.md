# Laporan Proyek Machine Learning - Muhammad Fadhil Diandra

## Project Overview

Dalam era digital saat ini, platform streaming seperti Netflix, Disney+, dan Amazon Prime menawarkan ribuan judul film dan serial. Pengguna kerap kesulitan memilih konten yang sesuai preferensi mereka. Sistem rekomendasi hadir sebagai solusi penting untuk menyaring konten berdasarkan preferensi dan perilaku pengguna. Dengan menganalisis pola rating pengguna terhadap film, sistem ini dapat memprediksi film yang kemungkinan besar akan disukai pengguna di masa depan.

Salah satu pendekatan populer adalah collaborative filtering, yaitu mempelajari interaksi pengguna dan menemukan kesamaan perilaku dengan pengguna lain. Dalam proyek ini, pendekatan matrix factorization menggunakan algoritma Singular Value Decomposition (SVD) diterapkan untuk memprediksi rating film yang belum ditonton pengguna. SVD dipilih karena kemampuannya mengatasi matriks yang sparse dan memberikan hasil prediksi yang akurat [2].

Model ini dibangun menggunakan data dari MovieLens yang menyediakan lebih dari 100.000 rating pengguna terhadap film. Output model berupa prediksi rating akan digunakan untuk memberikan rekomendasi film yang bersifat personal dan relevan bagi setiap pengguna. Studi oleh Netflix menunjukkan bahwa lebih dari 80% aktivitas menonton berasal dari sistem rekomendasi otomatis mereka [3], memperkuat pentingnya sistem seperti ini dalam meningkatkan pengalaman pengguna.

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
- movie.csv (27278 baris) berisi kolom movieId, title, dan genres.
- rating.csv (2.003.488 baris) berisi kolom userId, movieId, rating, dan timestamp

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

## Data Preparation
- Sebelum modeling, kita perlu menggabungkan data rating dengan data film agar tiap entri rating juga membawa informasi judul dan genre:
data_merged = pd.merge(ratings, movies, on='movieId')
data_merged.head()
pd.merge(ratings, movies, on='movieId') menggabungkan baris‐baris dari ratings dan movies berdasarkan kolom movieId yang sama.
-Menggunakan TF-IDF Vectorizer untuk merepresentasikan fitur genres.
Menghitung cosine similarity antar film.

Hasilnya, data_merged memiliki kolom:
[userId, movieId, rating, timestamp, title, genres]

- Proses vektorisasi terhadap kolom 'genres' menggunakan metode TF-IDF (Term Frequency-Inverse Document Frequency). Teknik ini digunakan untuk mengubah nilai teks pada genre menjadi representasi numerik dalam bentuk matriks. Representasi ini memungkinkan pengukuran kemiripan antar film berdasarkan kesamaan genre.

- Proses ini dilakukan menggunakan TfidfVectorizer dari library sklearn.feature_extraction.text dengan parameter stop_words='english' untuk menghapus kata-kata umum dalam Bahasa Inggris yang tidak bermakna penting (seperti "the", "and", dll).

- Menggunakan Reader() dari library Surprise untuk menetapkan skala rating.

- Dataset dibagi menjadi trainset dan testset (80:20).

## Exploratory Data Analysis (EDA):

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


## Modeling: Collaborative Filtering dengan SVD
Algoritma: Singular Value Decomposition dari pustaka Surprise.

Pelatihan model dilakukan pada trainset, lalu diuji pada testset.
Evaluasi performa:
- RMSE: 0.8193
- MAE: 0.6265

Berikut hasil yang ditampilkan rekomendasi top-N untuk userId=1. :

Rekomendasi untuk User 1:
 ['Terminator 2: Judgment Day (1991)' 'Die Hard (1988)'
 'E.T. the Extra-Terrestrial (1982)' 'Ran (1985)'
 'Dead Poets Society (1989)' 'Sling Blade (1996)'
 'Untouchables, The (1987)'
 'Star Wars: Episode I - The Phantom Menace (1999)'
 'Lost Boys, The (1987)'
 "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)"]


## Modeling: Content-Based Filtering (CBF)
Rekomendasi diberikan berdasarkan film yang mirip secara genre, berikut hasil yang ditampilkan :
CBF - Rekomendasi berdasarkan 'Toy Story (1995)':
1. Antz (1998)
2. Toy Story 2 (1999)
3. Adventures of Rocky and Bullwinkle, The (2000)
4. Emperor's New Groove, The (2000)
5. Monsters, Inc. (2001)
6. DuckTales: The Movie - Treasure of the Lost Lamp (1990)
7. Wild, The (2006)
8. Shrek the Third (2007)
9. Tale of Despereaux, The (2008)
10. Asterix and the Vikings (Astérix et les Vikings) (2006)

Menggunakan Top N recomendation merekomendasikan film yang mirip dengan toy story sebanyak 10 film.

##  Baseline Predictive Model
Strategi: memprediksi rating menggunakan rata-rata rating film.
Evaluasi performa:
- RMSE: 0.9332593915343897
- MAE: 0.7259228114918184

Error lebih tinggi dibanding model SVD, namun berguna sebagai acuan dasar.

## Evaluation
Evaluasi dilakukan menggunakan metrik yang sesuai untuk masing-masing jenis model, yaitu RMSE (Root Mean Square Error) dan MAE (Mean Absolute Error) untuk model prediktif seperti SVD dan Baseline, serta Precision@10 dan Recall@10 untuk model non-prediktif seperti Content-Based Filtering.

1. SVD (Singular Value Decomposition)
Model SVD menunjukkan performa terbaik dalam memprediksi rating numerik. Hasil evaluasinya adalah sebagai berikut:

- RMSE: 0.8193

- MAE: 0.6265

2. Baseline Model
Model ini digunakan sebagai pembanding dasar (benchmark) dan menghasilkan performa yang lebih rendah dibandingkan SVD:

- RMSE: 0.9333

- MAE: 0.7259

3. Content-Based Filtering (CBF)
Karena Content-Based Filtering tidak memprediksi nilai rating secara langsung, model ini dievaluasi menggunakan metrik Precision@10 dan Recall@10, yang mengukur relevansi rekomendasi terhadap pengguna. Hasil evaluasinya adalah:

- Precision@10: 0.0000

- Recall@10: 0.0000

Nilai evaluasi yang rendah pada CBF menunjukkan bahwa model ini belum mampu menghasilkan rekomendasi yang relevan untuk pengguna dalam top-10 item yang disarankan.

# Conclusion
- SVD memiliki nilai error yang paling rendah, menunjukkan kemampuannya dalam memprediksi rating pengguna dengan baik.

- Baseline Model memberikan hasil yang kurang akurat dibandingkan SVD.

- CBF tidak dievaluasi menggunakan RMSE/MAE karena bersifat non-prediktif
