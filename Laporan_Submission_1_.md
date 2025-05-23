# Laporan Proyek Machine Learning - Muhammad Fadhil Diandra

## Domain Proyek

### Latar Belakang
Diabetes merupakan salah satu penyakit tidak menular yang menjadi ancaman kesehatan global.  
Berdasarkan data dari World Health Organization (WHO), pada tahun 2021, terdapat sekitar 537 juta orang dewasa yang hidup dengan diabetes, dan jumlah ini diperkirakan akan terus meningkat [1]. 
Di Indonesia, data dari International Diabetes Federation (IDF) menunjukkan bahwa pada tahun 2021 terdapat sekitar 19,5 juta penderita diabetes, menjadikan Indonesia sebagai salah satu negara dengan jumlah penderita tertinggi di dunia [2].

Masalah utama dari diabetes adalah sifatnya yang kronis dan seringkali tidak terdeteksi sejak awal. Oleh karena itu, deteksi dini sangat penting untuk mencegah komplikasi lebih lanjut. 
Dengan kemajuan teknologi dan ketersediaan data medis, pendekatan machine learning dapat digunakan untuk membangun model prediktif guna mengidentifikasi potensi diabetes berdasarkan data medis pasien.

### Mengapa Perlu diselesaikan ?
- Deteksi dini diabetes memungkinkan penanganan yang lebih cepat dan tepat.
- Mengurangi beban sistem kesehatan dengan pencegahan komplikasi jangka panjang.
- Meningkatkan kesadaran akan faktor risiko terhadap masyarakat umum.

### Referensi 
[1] World Health Organization. (2021). Diabetes. https://www.who.int/news-room/fact-sheets/detail/diabetes
[2] International Diabetes Federation. (2021). IDF Diabetes Atlas, 10th edition.

## Business Understanding

### Problem Statements

- Bagaimana memprediksi apakah seseorang menderita diabetes berdasarkan data medis seperti kadar glukosa, tekanan darah, dan usia?
- Fitur mana yang paling berpengaruh dalam menentukan prediksi diabetes?

### Goals

- Membangun model klasifikasi yang dapat memprediksi kondisi diabetes seseorang.
- Menganalisis pentingnya fitur untuk memberikan wawasan kepada praktisi medis.

### Solution statements
- Menggunakan dua algoritma machine learning yaitu logistic regression dan random forest untuk lanjutan
- Mengukur kinerja model dengan metrik: Accuracy, Precision, Recall, dan F1-score.
- Melakukan hyperparameter tuning pada model Random Forest untuk peningkatan performa.

## Data Understanding
Dataset yang digunakan adalah Pima Indians Diabetes Database dari Kaggle:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Variabel-variabel pada dataset adalah sebagai berikut:
- Pregnancies: Jumlah kehamilan
- Glucose: Kadar glukosa dalam darah
- BloodPressure: Tekanan darah diastolik (mm Hg)
- SkinThickness: Ketebalan lipatan kulit triceps (mm)
- Insulin: Kadar insulin dalam serum (mu U/ml)
- BMI: Indeks massa tubuh (berat badan dalam kg dibagi kuadrat tinggi badan dalam m)
- DiabetesPedigreeFunction: Fungsi silsilah diabetes
- Age: Usia pasien (tahun)
- Outcome: 0 = tidak diabetes, 1 = diabetes

### Visualisasi Data
Untuk memahami data lebih lanjut, dilakukan beberapa visualisasi berikut:
- Distribusi Kelas Target (Outcome): Barplot menunjukkan data tidak seimbang (lebih banyak yang tidak diabetes).
- Distribusi Fitur Numerik: Histogram menunjukkan fitur seperti Insulin, Glucose, dan BMI memiliki skewness.
- Nilai 0 Tidak Wajar: Countplot menunjukkan adanya nilai 0 pada fitur medis yang seharusnya tidak nol seperti Glucose, BloodPressure, SkinThickness, Insulin, dan BMI.
- Heatmap Korelasi: Menunjukkan bahwa Glucose, BMI, dan Age memiliki korelasi cukup kuat terhadap Outcome.

## Data Preparation
- Mengganti nilai 0 pada fitur medis (Glucose, BloodPressure, SkinThickness, Insulin, dan BMI) dengan nilai median kolom tersebut.
- Melakukan standardisasi data menggunakan StandardScaler.
- Membagi data menjadi training set dan testing set (80:20).

Karena : Nilai 0 dapat menurunkan kualitas model karena secara medis tidak mungkin bernilai nol. Standardisasi diperlukan untuk membuat algoritma bekerja optimal, terutama pada model yang sensitif terhadap skala seperti Logistic Regression.

## Modeling
Dua algoritma digunakan:
1. Logistic Regression (Baseline)
- Parameter default digunakan.
- Kelebihan: Cepat, mudah diinterpretasi, cocok untuk baseline.
- Kekurangan: Kurang cocok jika hubungan fitur dengan target tidak linear.
2. Random Forest Classifier
- Parameter awal: n_estimators=100, max_depth=None
- Setelah hyperparameter tuning: n_estimators=150, max_depth=8
- Kelebihan: Akurasi tinggi, robust terhadap outlier, tidak memerlukan scaling
- Kekurangan: Lebih kompleks, sulit diinterpretasi

### Model Terbaik
Setelah model baseline dibuat, dilakukan hyperparameter tuning pada model Random Forest menggunakan GridSearchCV. 
Parameter yang dituning meliputi:
- n_estimators: [100, 150, 200]
- max_depth: [None, 6, 8, 10]
Tujuan dari tuning ini adalah untuk mencari kombinasi parameter terbaik guna meningkatkan performa model. 
Hasil tuning menunjukkan bahwa kombinasi `n_estimators=150` dan `max_depth=8` memberikan performa terbaik pada data validasi.
 
## Evaluation
Metrik yang digunakan:
- Accuracy: Proporsi prediksi benar dari seluruh prediksi.
- Precision: Proporsi prediksi positif yang benar-benar positif.
- Recall: Proporsi data positif yang berhasil diprediksi dengan benar.
- F1-score: Harmonik rata-rata precision dan recall.

### Hasil Evaluasi
1. Logistik Regression
-
-
3. Random Forest
-
- 
