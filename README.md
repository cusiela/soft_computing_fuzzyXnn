# 🧠 NeuroFuzzy Medical Diagnosis System

> Sistem Diagnosis Medis menggunakan Hybrid Fuzzy Expert System + Neural Network untuk Prediksi Risiko Stroke

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![System Preview](docs/preview.png)

## 📋 Daftar Isi

- [Fitur](#-fitur)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi](#-instalasi)
- [Penggunaan](#-penggunaan)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [API Endpoints](#-api-endpoints)
- [Testing](#-testing)
- [Metrik Evaluasi](#-metrik-evaluasi)
- [Dokumentasi](#-dokumentasi)

## ✨ Fitur

### 🎯 Core Features
- **Fuzzy Expert System**: Implementasi lengkap dari scratch dengan membership functions (triangular, trapezoidal, gaussian, sigmoid)
- **Neural Network**: Multi-layer perceptron dengan backpropagation dari scratch
- **Hybrid Model**: Kombinasi Fuzzy Logic + Neural Network untuk akurasi lebih tinggi
- **Web Interface**: Dashboard interaktif dengan visualisasi real-time

### 📊 Analytics & Visualization
- **EDA Dashboard**: Exploratory Data Analysis dengan charts interaktif
- **Performance Metrics**: Sensitivity, Specificity, AUC-ROC, Accuracy, F1-Score
- **ROC Curve**: Visualisasi ROC curve dengan Chart.js
- **Confusion Matrix**: Tampilan matriks konfusi interaktif

### 💻 Technical Features
- 100% Python dari scratch (tanpa sklearn untuk algoritma utama)
- Modular & well-documented code
- Comprehensive error handling
- Unit testing dengan 30+ test cases

## 📁 Struktur Proyek

```
medical_diagnosis_system/
├── app.py                    # Flask main application
├── models/
│   ├── __init__.py          # Package initialization
│   ├── fuzzy.py             # Fuzzy Logic implementation
│   ├── neural.py            # Neural Network implementation
│   ├── metrics.py           # Evaluation metrics
│   └── preprocessing.py     # Data preprocessing
├── data/
│   ├── raw/
│   │   └── stroke_data.csv  # Original dataset
│   └── processed/           # Cleaned datasets
├── notebooks/
│   ├── EDA.ipynb           # Exploratory Data Analysis
│   └── manual_calc.ipynb   # Manual calculations
├── static/
│   ├── css/
│   │   └── style.css       # Stylesheet
│   └── js/
│       └── app.js          # Frontend JavaScript
├── templates/
│   └── index.html          # Main HTML template
├── tests/
│   └── test_models.py      # Unit tests
├── docs/
│   ├── teori.md            # Theory documentation
│   └── perhitungan.md      # Manual calculations
├── requirements.txt
└── README.md
```

## 🚀 Instalasi

### Prerequisites
- Python 3.8 atau lebih baru
- pip (Python package manager)

### Step-by-Step Installation

1. **Clone repository**
```bash
git clone https://github.com/yourusername/medical_diagnosis_system.git
cd medical_diagnosis_system
```

2. **Buat virtual environment (opsional tapi direkomendasikan)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Jalankan aplikasi**
```bash
python app.py
```

5. **Buka browser**
```
http://localhost:5000
```

## 📖 Penggunaan

### Web Interface

1. **Input Data Pasien**
   - Age: Usia pasien (0-120 tahun)
   - BMI: Body Mass Index (10-60 kg/m²)
   - Glucose: Rata-rata level glukosa (50-300 mg/dL)
   - Hypertension: Riwayat hipertensi (Yes/No)
   - Heart Disease: Riwayat penyakit jantung (Yes/No)

2. **Klik "Analyze Risk"**

3. **Lihat Hasil**
   - Stroke Risk Probability (%)
   - Risk Level (Very Low → Very High)
   - Severity Score (0-10)
   - Detailed breakdowns (Fuzzy vs Neural Network)

### Python API

```python
from models import create_stroke_fuzzy_system, create_hybrid_model

# Inisialisasi sistem
fuzzy_system = create_stroke_fuzzy_system()

# Prediksi dengan fuzzy only
result = fuzzy_system.predict(
    age=67,
    glucose=228.69,
    bmi=36.6,
    hypertension=1,
    heart_disease=1
)

print(f"Risk: {result['stroke_risk_percentage']}%")
print(f"Level: {result['risk_level']}")
print(f"Severity: {result['severity_score']}/10")
```

### Hybrid Model Training

```python
from models import create_stroke_fuzzy_system, create_hybrid_model

# Data training
train_data = [
    {'age': 75, 'avg_glucose_level': 200, 'bmi': 32, 
     'hypertension': 1, 'heart_disease': 1},
    {'age': 30, 'avg_glucose_level': 90, 'bmi': 22,
     'hypertension': 0, 'heart_disease': 0}
]
train_labels = [1, 0]  # 1 = stroke, 0 = no stroke

# Buat dan train hybrid model
fuzzy_system = create_stroke_fuzzy_system()
hybrid = create_hybrid_model(fuzzy_system)
history = hybrid.fit(train_data, train_labels, epochs=100)

# Prediksi
result = hybrid.predict_single(train_data[0])
```

## 🏗 Arsitektur Sistem

### Fuzzy Expert System

```
Input Variables:          Output Variables:
├── age (0-100)          ├── stroke_risk (0-100%)
├── glucose (50-300)     │   ├── very_low
├── bmi (10-60)          │   ├── low
├── hypertension (0-1)   │   ├── moderate
└── heart_disease (0-1)  │   ├── high
                         │   └── very_high
                         └── severity (0-10)
                             ├── mild
                             ├── moderate
                             └── severe
```

### Neural Network Architecture

```
Input Layer     Hidden Layer 1    Hidden Layer 2    Output Layer
   (n)              (32)              (16)             (1)
    ○                ○                 ○                
    ○ ───────────── ○ ───────────── ○ ──────────────── ○
    ○       ReLU    ○     ReLU      ○     Sigmoid      
    ○                ○                 ○                
    ○                ○                                  
```

### Hybrid Pipeline

```
Raw Input → Fuzzy System → Feature Extraction → Neural Network → Combined Output
                ↓                                      ↓
          Fuzzy Output                          NN Probability
                ↓                                      ↓
                └──────────────────────────────────────┘
                                  ↓
                        Final Prediction
                   (0.4×Fuzzy + 0.6×NN)
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/predict` | POST | Get stroke risk prediction |
| `/metrics` | GET | Get model performance metrics |
| `/eda` | GET | Get EDA results |
| `/fuzzy/rules` | GET | Get fuzzy rules list |
| `/fuzzy/membership/<var>` | GET | Get membership function data |
| `/api/health` | GET | System health check |

### Example API Call

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 67,
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "hypertension": 1,
    "heart_disease": 1
  }'
```

Response:
```json
{
  "stroke_probability": 78.5,
  "risk_level": "High",
  "severity_score": 7.2,
  "severity_level": "Severe",
  "fuzzy_risk": 82.3,
  "nn_probability": 75.9,
  "confidence": 85.0,
  "prediction": 1
}
```

## 🧪 Testing

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Specific Test Module

```bash
python tests/test_models.py
```

### Test Coverage

```bash
pip install pytest-cov
pytest --cov=models tests/
```

### Test Cases Overview

| Module | Test Cases | Description |
|--------|------------|-------------|
| Fuzzy | 9 tests | Membership functions, fuzzification, rules |
| Neural | 5 tests | Forward pass, backprop, training |
| Metrics | 7 tests | Accuracy, sensitivity, specificity, AUC |
| Preprocessing | 6 tests | Data cleaning, encoding, scaling |
| Integration | 1 test | End-to-end hybrid model |

## 📈 Metrik Evaluasi

### Performance Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Sensitivity | TP/(TP+FN) | True Positive Rate |
| Specificity | TN/(TN+FP) | True Negative Rate |
| Precision | TP/(TP+FP) | Positive Predictive Value |
| F1-Score | 2×(P×R)/(P+R) | Harmonic mean of P and R |
| AUC-ROC | Area under ROC curve | Overall discrimination ability |

### Interpretasi AUC

| AUC Range | Interpretation |
|-----------|----------------|
| 0.5 - 0.6 | No discrimination |
| 0.6 - 0.7 | Poor |
| 0.7 - 0.8 | Acceptable |
| 0.8 - 0.9 | Excellent |
| 0.9 - 1.0 | Outstanding |

## 📚 Dokumentasi

- **[Teori](docs/teori.md)**: Penjelasan lengkap tentang Fuzzy Logic dan Neural Network
- **[Perhitungan Manual](docs/perhitungan.md)**: Contoh perhitungan step-by-step

## 🛠 Dependencies

```
Flask>=2.0.0
```

> **Note**: Semua algoritma utama (Fuzzy Logic, Neural Network, Metrics) diimplementasikan dari scratch tanpa sklearn.

## 📄 Dataset

Dataset stroke prediction dengan atribut:
- `id`: Unique identifier
- `gender`: Male/Female
- `age`: Age of patient
- `hypertension`: 0/1
- `heart_disease`: 0/1
- `ever_married`: Yes/No
- `work_type`: Type of work
- `Residence_type`: Urban/Rural
- `avg_glucose_level`: Average glucose level
- `bmi`: Body Mass Index
- `smoking_status`: Smoking status
- `stroke`: Target variable (0/1)

## ⚠️ Disclaimer

> **PENTING**: Sistem ini dibuat untuk tujuan **edukasi dan penelitian** saja. TIDAK dimaksudkan untuk penggunaan klinis atau diagnosis medis yang sebenarnya. Selalu konsultasikan dengan tenaga medis profesional untuk keputusan kesehatan.

## 📝 License

MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 👨‍💻 Author

Medical Diagnosis System - Soft Computing Project

---

<p align="center">
  Made with ❤️ using Python, Flask, and pure algorithms from scratch
</p>
