# Yapay Zeka Ödevi: SMS/E-Posta Spam Tespiti ve Sınıflandırma

Bu proje, SMS ve e-posta metinleri üzerinden spam tespiti gerçekleştirmek için geliştirilmiş bir makine öğrenmesi uygulamasıdır. Proje kapsamında metin ön işleme (NLP), TF-IDF vektörizasyonu, farklı sınıflandırma modellerinin eğitimi, hiperparametre optimizasyonu (GridSearchCV), topluluk öğrenmesi (Ensemble Learning) ve kapsamlı model değerlendirmesi (Confusion Matrix, ROC-AUC) yer almaktadır.

Detaylı analiz ve akademik anlatım için [RAPOR.md](RAPOR.md) dosyasını inceleyebilirsiniz.

---

## 📌 Proje Özeti ve Model Başarıları

Modeller **1.115 adet test mesajı** üzerinde test edilmiş olup elde edilen metrikler:

| Algoritma | Baseline Doğruluk | Optimize Doğruluk | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Support Vector Machine (SVM)** | **%97.67** | **%98.92** | **%98.58** | **%93.29** | **%95.86** | **0.9910** |
| **Logistic Regression** | %98.12 | %98.74 | %95.92 | %94.63 | %95.27 | **0.9930** |
| **Multinomial Naive Bayes** | %98.74 | %98.39 | **%99.25** | %88.59 | %93.62 | 0.9862 |
| **Soft Voting Ensemble** | --- | %98.74 | %97.20 | %93.29 | %95.21 | 0.9923 |

> 🏆 **En Başarılı Model:** Support Vector Machine (SVM), test kümesinde **%98.92 Doğruluk** ve **%95.86 F1-Skoru** ile en yüksek performansı göstermiştir.

---

## 📊 Performans Grafikleri

### 1. Karmaşıklık Matrisleri (Confusion Matrices)
Tüm modellerin Doğru Pozitif (TP), Yanlış Pozitif (FP), Doğru Negatif (TN) ve Yanlış Negatif (FN) dağılımları:

![Confusion Matrices](confusion_matrices.png)

### 2. ROC ve AUC Eğrileri
Modellerin sınıf ayrım güçlerini gösteren ROC eğrileri:

![ROC Curves](roc_curves.png)

---

## 🚀 Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/sinansamiil/yapay_zeka_odev.git
cd yapay_zeka_odev
```

### 2. Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Modeli Eğitin ve Test Edin
```bash
python spam_classifier.py
```

Betiği çalıştırdığınızda:
- Veri seti otomatik olarak indirilir ve temizlenir,
- Naive Bayes, SVM, Logistic Regression ve Voting Ensemble modelleri eğitilir,
- Hiperparametre optimizasyonu (GridSearchCV) uygulanır,
- Metrik sonuçları terminalde ve `metrics_summary.json` dosyasında raporlanır,
- `confusion_matrices.png` ve `roc_curves.png` grafikleri üretilir,
- Yeni örnek mesajlar üzerinde canlı tahmin testi yapılır.

---

## 📁 Proje Dosya Yapısı

```
├── spam_classifier.py       # Veri işleme, model eğitimi, optimizasyon ve test betiği
├── RAPOR.md                 # Sorulara cevap veren ayrıntılı ödev raporu
├── requirements.txt         # Gerekli Python kütüphaneleri
├── confusion_matrices.png   # Modellerin karmaşıklık matrisi görseli
├── roc_curves.png           # Modellerin ROC eğrileri görseli
├── metrics_summary.json     # Test metriklerinin JSON formatındaki çıktısı
└── README.md                # Proje tanıtım ve çalıştırma rehberi
```
