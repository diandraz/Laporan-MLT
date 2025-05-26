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

- Dataset memiliki 768 baris dan 9 kolom, yang berarti berisi data 768 pasien wanita keturunan Indian Pima, dengan 8 fitur dan 1 target (Outcome).
- Terdiri dari 7 kolom integer dan 2 kolom float.
- Semua kolom memiliki nilai non-null secara teknis (tidak ada NaN), namun perlu evaluasi lanjutan terhadap nilai nol yang tidak logis.
- Nilai Nol Tidak Logis (Implicit Missing Values)
Meskipun tidak ada missing value secara eksplisit, beberapa fitur mengandung nilai nol yang secara medis tidak mungkin terjadi. Berikut fitur-fitur yang terpengaruh:
 -Glucose memiliki 5 nilai nol.
 -BloodPressure memiliki 35 nilai nol.
 -SkinThickness memiliki 227 nilai nol.
 -Insulin memiliki 374 nilai nol.
 -BMI memiliki 11 nilai nol.

Nilai-nilai nol ini kemungkinan besar mewakili data yang hilang, dan perlu diimputasi (diganti dengan nilai median, mean, atau strategi lainnya). Sebaliknya, kolom Pregnancies memang wajar memiliki nol karena pasien bisa saja belum pernah hamil.

- Distribusi Label (Outcome)
Dari 768 data, terdapat:

 -500 data dengan label 0 (tidak menderita diabetes)
 -268 data dengan label 1 (menderita diabetes)

Artinya, kelas target tidak seimbang. Sekitar 65% data merupakan kelas negatif (tidak diabetes), dan 35% merupakan kelas positif (diabetes). Ketidakseimbangan ini perlu diperhatikan dalam proses training model agar tidak bias terhadap mayoritas kelas.

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
Model terbaik dipilih berdasarkan evaluasi performa dan keseimbangan metrik antar kelas. Model Random Forest Classifier dengan parameter default (tanpa tuning) dipilih sebagai model akhir karena memberikan performa terbaik secara keseluruhan.
 
## Evaluation
Metrik yang digunakan:
- Accuracy: Proporsi prediksi benar dari seluruh prediksi.
- Precision: Proporsi prediksi positif yang benar-benar positif.
- Recall: Proporsi data positif yang berhasil diprediksi dengan benar.
- F1-score: Harmonik rata-rata precision dan recall.

### Hasil Evaluasi
1. Logistik Regression
- Accuracy: 76%
- Precision (class 1 / diabetes): 66%
- Recall (class 1 / diabetes): 54% 
- F1-score (class 1 / diabetes): 59%
2. Random Forest
- Accuracy: 78%
- Precision (class 1 / diabetes): 73%
- Recall (class 1 / diabetes): 59%
- F1-score (class 1 / diabetes): 65%
