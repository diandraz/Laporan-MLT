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
 - 500 data dengan label 0 (tidak menderita diabetes)
 - 268 data dengan label 1 (menderita diabetes)

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
- Mengganti Nilai Nol dengan Median
- Memisahkan Fitur dan Target
- Membagi data menjadi training set dan testing set (80:20).
- Melakukan standardisasi data menggunakan StandardScaler.

Karena : Nilai 0 dapat menurunkan kualitas model karena secara medis tidak mungkin bernilai nol. Standardisasi diperlukan untuk membuat algoritma bekerja optimal, terutama pada model yang sensitif terhadap skala seperti Logistic Regression.

## Modeling
Dua algoritma digunakan:
1. Logistic Regression (Baseline)
- Parameter default yang digunakan random_state=42.
- Cara Kerja:
Logistic Regression adalah algoritma klasifikasi linier yang memodelkan hubungan antara fitur dan probabilitas suatu kelas target. Algoritma ini menghitung kombinasi linier dari semua fitur dan kemudian mengubah hasilnya menggunakan fungsi sigmoid untuk menghasilkan nilai probabilitas antara 0 dan 1. Prediksi dibuat berdasarkan apakah probabilitas tersebut melewati ambang batas tertentu (biasanya 0.5). Logistic Regression sangat cocok untuk klasifikasi biner dan dapat memberikan interpretasi koefisien fitur terhadap prediksi.
- Kelebihan :
     - Cepat dan efisien untuk dataset kecil hingga sedang.
     - Mudah diinterpretasi (koefisien mencerminkan pengaruh fitur).
     - Cocok sebagai baseline.
- Kekurangan :
     - Tidak dapat menangani hubungan non-linear antar fitur secara langsung.
     - Performa menurun jika data tidak terdistribusi secara linier.
  
2. Random Forest Classifier
- Parameter awal: n_estimators=100, max_depth=None, random_state=42
- Cara Kerja:
Random Forest adalah algoritma ensemble berbasis pohon keputusan. Ia membangun banyak pohon (decision tree) secara acak dari subset data dan subset fitur yang berbeda, lalu menggabungkan prediksinya melalui voting (untuk klasifikasi). Karena tidak semua data digunakan dalam satu pohon dan pemilihan fitur dilakukan secara acak, Random Forest mampu menangani overfitting lebih baik daripada pohon tunggal. Algoritma ini juga tidak memerlukan normalisasi data.
- Kelebihan :
     - Akurasi tinggi dan stabil.
     - Tidak sensitif terhadap outlier dan scaling.
     - Dapat menangani fitur kategorik dan numerik secara alami.
- Kekurangan :
     - Lebih kompleks dibanding model linier.
     - Sulit diinterpretasikan secara langsung karena banyak pohon

### Model Terbaik
Berdasarkan hasil evaluasi pada data uji, model Random Forest Classifier dengan parameter default menunjukkan performa yang lebih baik secara keseluruhan dibandingkan Logistic Regression. Hal ini ditunjukkan oleh akurasi yang lebih tinggi, serta metrik recall dan f1-score yang lebih baik dalam mendeteksi kasus diabetes (kelas positif).

Dengan demikian, Random Forest Classifier dipilih sebagai model akhir karena lebih mampu menangani kompleksitas data dan memberikan prediksi yang lebih seimbang antar kelas meskipun belum dilakukan tuning.
 
## Evaluation
Metrik yang digunakan:
- Accuracy: Proporsi prediksi benar dari seluruh prediksi.
- Precision: Proporsi prediksi positif yang benar-benar positif.
- Recall: Proporsi data positif yang berhasil diprediksi dengan benar.
- F1-score: Harmonik rata-rata precision dan recall.

### Hasil Evaluasi
Model Logistic Regression menghasilkan akurasi sebesar 71%, precision sebesar 60%, recall sebesar 50%, dan f1-score sebesar 55% pada kelas positif (penderita diabetes).
Model Random Forest Classifier, yang digunakan tanpa proses hyperparameter tuning, menunjukkan peningkatan performa dengan akurasi sebesar 76%, precision sebesar 68%, recall sebesar 59%, dan f1-score sebesar 63% pada kelas positif.

Tujuan utama dari proyek ini adalah untuk mendeteksi potensi diabetes secara dini berdasarkan data medis sederhana. Dengan demikian, model yang dibangun diharapkan dapat:

- Membantu tenaga medis membuat keputusan awal yang cepat dan berbasis data.

- Mengurangi keterlambatan diagnosis yang berpotensi memperparah kondisi pasien.

- Memberikan solusi yang efisien dan praktis untuk diterapkan dalam sistem pelayanan kesehatan.

Model yang dikembangkan berhasil menjawab permasalahan tersebut. Logistic Regression, sebagai model baseline, sudah menunjukkan performa awal yang cukup baik, namun masih kesulitan dalam mendeteksi penderita diabetes (recall hanya 50%). Sebaliknya, Random Forest menunjukkan kinerja yang lebih seimbang antara mengenali pasien sehat dan pasien yang benar-benar menderita diabetes. Dengan recall sebesar 59%, Random Forest mampu menangkap lebih banyak kasus positif, yang sangat penting dalam konteks medis.

Selain itu, meskipun model belum mencapai performa sempurna, hasil ini sudah cukup menunjukkan bahwa pendekatan machine learning dapat memberikan nilai tambah yang signifikan bagi pelayanan kesehatan, khususnya dalam mendukung deteksi dini penyakit kronis seperti diabetes.

Insight Akhir
Model Random Forest dipilih sebagai model terbaik karena memberikan performa paling stabil dan mendekati tujuan utama proyek, yaitu mendeteksi potensi diabetes secara akurat dan efisien. Dengan pengembangan lebih lanjut, seperti tuning parameter atau penambahan fitur medis yang lebih kompleks, model ini masih memiliki ruang untuk ditingkatkan.
