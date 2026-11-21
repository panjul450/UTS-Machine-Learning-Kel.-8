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
