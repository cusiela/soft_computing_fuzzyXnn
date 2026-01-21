# 🧠 Hybrid Fuzzy-Neural Network Stroke Diagnosis System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Active-success)

Sistem diagnosis stroke cerdas yang menggabungkan interpretabilitas **Fuzzy Logic** dengan kemampuan pembelajaran pola kompleks dari **Neural Network**. Proyek ini dibangun dari awal (*from scratch*) tanpa menggunakan library *black-box* untuk inti algoritmanya, bertujuan untuk memberikan transparansi dalam keputusan medis.

## 📋 Latar Belakang

Penyakit serebrovaskular, khususnya stroke, menempati peringkat kedua penyebab kematian tertinggi di dunia (World Stroke Organization, 2025). Di Indonesia, tren prevalensi stroke terus meningkat seiring dengan tingginya faktor risiko metabolik seperti hipertensi, diabetes, dan obesitas.

Tantangan utama dalam penerapan AI di bidang medis adalah masalah interpretabilitas. Model *Machine Learning* konvensional seringkali bekerja sebagai "kotak hitam". Sistem ini menawarkan solusi **Hibrid**:
1.  **Fuzzy Logic**: Menangani ketidakpastian data klinis (misalnya: definisi "usia tua" atau "glukosa tinggi") secara alami.
2.  **Neural Network**: Mempelajari pola non-linear yang kompleks dari data historis.
3.  **Weighted Combination**: Menggabungkan kedua output untuk menghasilkan prediksi risiko yang akurat namun tetap dapat dijelaskan.

---

## ✨ Fitur Utama

### 🎯 Core Features
- **Fuzzy Expert System**: Implementasi logika fuzzy dari scratch dengan berbagai fungsi keanggotaan (triangular, trapezoidal, gaussian, sigmoid). Sistem menangani proses fuzzifikasi, evaluasi aturan, hingga defuzzifikasi.
- **Neural Network**: Arsitektur Multi-layer Perceptron (MLP) dengan algoritma backpropagation yang dibangun dari dasar.
- **Hybrid Model**: Mekanisme kombinasi linear terbobot ($P_{hybrid} = w_f P_f + w_n P_n$) untuk memaksimalkan akurasi.
- **Web Interface**: Antarmuka berbasis web yang user-friendly untuk input data dan visualisasi hasil diagnosis secara real-time.

### 📊 Analytics & Visualization
- **EDA Dashboard**: Halaman khusus untuk *Exploratory Data Analysis* dengan grafik interaktif mengenai sebaran data pasien.
- **Performance Metrics**: Kalkulasi otomatis untuk Sensitivity, Specificity, AUC-ROC, Accuracy, dan F1-Score.
- **ROC Curve**: Visualisasi kurva ROC menggunakan Chart.js untuk evaluasi trade-off model.
- **Confusion Matrix**: Tampilan interaktif untuk melihat detail True Positive, False Positive, dll.

---

## 📸 Antarmuka Sistem

### 1. Halaman Diagnosis
Melakukan prediksi risiko stroke berdasarkan parameter klinis pasien secara real-time. Menampilkan kontribusi skor dari Fuzzy System dan Neural Network secara terpisah.

![Diagnosis Page](visualization_outputs/assesmen.png)


### 2. Analisis Data & Performa Model
Dashboard komprehensif yang menampilkan statistik dataset (ketidakseimbangan kelas) dan metrik evaluasi model seperti Confusion Matrix dan Kurva ROC.

![Performance Page](visualization_outputs/performa_model.png)


---

## ⚙️ Arsitektur & Logika Sistem

Sistem ini bekerja berdasarkan alur perhitungan berikut (seperti yang divalidasi dalam dokumen perhitungan manual):

1.  **Input Pasien**: Usia, Glukosa, BMI, Hipertensi, Penyakit Jantung.
2.  **Fuzzy Processing**:
    * Variabel dikonversi menjadi derajat keanggotaan ($\mu$).
    * Inferensi aturan (contoh: *IF Age is Old AND Hypertension is Yes THEN Risk is Very High*).
    * Defuzzifikasi menggunakan metode Centroid.
3.  **Neural Network Processing**:
    * Input dinormalisasi dan diproses melalui *Hidden Layers* dengan aktivasi ReLU.
    * Output layer menggunakan aktivasi Sigmoid untuk probabilitas.
4.  **Hybrid Aggregation**:
    * $$P_{hybrid} = (0.1 \times P_{fuzzy}) + (0.9 \times P_{neural})$$
    * *Bobot ditentukan berdasarkan eksperimen pada dataset tidak seimbang.*
5.  **Keputusan Akhir**:
    * Menggunakan *Optimal Threshold* (0.40) berdasarkan Youden's J Statistic untuk memprioritaskan Sensitivitas (meminimalkan False Negative).

---

## 🚀 Step-by-Step Installation

Pastikan Anda telah menginstal Python 3.8 atau lebih baru.

1. **Clone repository**
   ```bash
   git clone [https://github.com/yourusername/medical_diagnosis_system.git](https://github.com/cusiela/soft_computing_fuzzyXnn.git)]
   cd medical_diagnosis_system

2. **Buat venv**
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate

3. **Install dependencies**
pip install -r requirements.txt

5. **Jalankan aplikasi**
python app.py

7. **Buka browser**
http://localhost:5000

## 📖 Cara Penggunaan

### 🩺 Melakukan Diagnosis

1. **Buka halaman Diagnosis** pada web interface.
2. **Masukkan data klinis pasien** pada form *Assesmen Pasien*:
   - **Age**: Usia pasien (Range: 0-120 tahun).
   - **BMI**: Body Mass Index (Range: 10-60 kg/m²).
   - **Glucose**: Rata-rata level glukosa darah (Range: 50-300 mg/dL).
   - **Hypertension**: Geser *toggle* jika pasien memiliki riwayat hipertensi.
   - **Heart Disease**: Geser *toggle* jika pasien memiliki riwayat penyakit jantung.
3. Klik tombol **Analyze Risk** (Menganalisa...).
4. Sistem akan menampilkan **Hasil Diagnosis** di panel kanan:
   - **Stroke Risk Probability**: Persentase kemungkinan stroke.
   - **Risk Level**: Kategori risiko (Low, Moderate, High, Very High).
   - **Breakdown**: Skor terpisah untuk Fuzzy dan Neural Network untuk transparansi.

### 📊 Melihat Analisis

1. Navigasi ke menu **Analisis** atau **Performa** melalui navigasi bar.
2. Anda dapat melihat:
   - Distribusi data pelatihan.
   - Metrik validasi model saat ini (Akurasi, Sensitivitas, AUC-ROC, dll).
