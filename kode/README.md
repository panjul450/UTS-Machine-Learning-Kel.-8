# Kode Sumber UTS Machine Learning - Kelompok 8

Direktori ini berisi seluruh skrip mandiri untuk mereproduksi hasil eksperimen dan angka-angka pada laporan (Tasks 1–3).

## Lingkungan & Kebutuhan Sistem
- Python 3.12+
- Dependensi utama: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`, `shap`

### Instalasi Dependensi
```bash
pip install -r ../requirements.txt
```

## Menjalankan Task 1 (Delivery Time Estimation - Regresi)
Satu perintah untuk menjalankan pipeline regresi kuantil, validasi conformal, dan menghasilkan file prediksi:
```bash
python task1_regresi.py
```
Output luaran:
- Visualisasi evaluasi di `Task 1/figures/`
- File prediksi submission: `../prediksi_regresi.csv` (60.000 baris, $p90 \ge p50$)

## Menjalankan Task 2 (Fraud Detection - Klasifikasi & Optimasi Biaya)
Satu perintah untuk menjalankan pipeline audit kebocoran data, rekayasa fitur perilaku kausal, komparasi model imbalanced (LightGBM, CatBoost, XGBoost), kalibrasi probabilitas, aturan keputusan berbasis biaya Bayes, serta menghasilkan file prediksi:
```bash
python task2_klasifikasi.py
```
Output luaran:
- Visualisasi evaluasi resolusi tinggi di `Task 2/figures/` (`fig1` s.d. `fig8`)
- File prediksi submission: `../prediksi_klasifikasi.csv` dan `../Task 2/prediksi_klasifikasi.csv` (106.674 baris, kolom `id_transaksi,prob,tandai`, $0 \le \text{prob} \le 1$, $\text{tandai} \in \{0, 1\}$)
