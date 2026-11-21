# UTS Machine Learning - Kelompok 8

Repositori ini memuat implementasi *honest and reproducible Machine Learning pipelines* untuk Ujian Tengah Semester (UTS) Mata Kuliah Machine Learning, mencakup analisis skala besar (~940.000 total baris data) pada tiga domain permasalahan utama:

1. **Task 1: Delivery Time Estimation (Regresi Kuantil & Conformal Prediction)**
2. **Task 2: Fraud Detection (Klasifikasi Imbalanced & Optimasi Biaya)**
3. **Task 3: Customer Segmentation (Clustering, Stabilitas ARI & Deteksi Anomali)**

---

## 📁 Struktur Repositori

Sesuai instruksi pengumpulan tugas, repositori ditata dengan format baku berikut:

```text
├── Instruksi/
│   ├── kamus_data.md          # Definisi fitur dan batasan operasional data
│   └── UTS (...).docx         # Panduan teknis & rubrik penilaian UTS
├── Task 1/
│   ├── figures/               # Gambar grafik evaluasi resolusi tinggi
│   ├── Regresi.ipynb          # Notebook terstruktur (10 tahap end-to-end)
│   └── prediksi_regresi.csv   # Salinan hasil prediksi Task 1
├── kode/
│   ├── README.md              # Petunjuk eksekusi mandiri untuk reproduksi angka
│   └── task1_regresi.py       # Skrip satu-perintah untuk mereproduksi Task 1
├── .gitignore                 # Konfigurasi ignore (data besar >100MB, venv, cache)
├── requirements.txt           # Daftar dependensi pustaka Python
├── prediksi_regresi.csv       # File submission Task 1 (id_paket, p50, p90)
├── prediksi_klasifikasi.csv   # [Task 2] Submission deteksi fraud
├── cluster.csv                # [Task 3] Submission segmentasi pelanggan
├── kontribusi.md              # Matriks kontribusi anggota kelompok
├── laporan.pdf                # Laporan tertulis resmi (maks 18 halaman)
└── README.md                  # Dokumentasi utama repositori ini
```

---

## 🚀 Setup Lingkungan & Instalasi

Pastikan menggunakan **Python 3.10+** (disarankan Python 3.12).

### 1. Kloning Repositori
```bash
git clone https://github.com/IkinInDisguise/UTS-Machine-Learning-Kel.-8.git
cd UTS-Machine-Learning-Kel.-8
```

### 2. Membuat Lingkungan Virtual & Dependensi
```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Mengunduh Dataset
Dataset mentah (`data_uts_ml.zip`) diunduh melalui tautan resmi yang disediakan pada instruksi tugas, kemudian diekstrak ke dalam folder `Dataset/`:
- `Dataset/regresi_train.csv` (Januari–Agustus 2025, ~100 MB)
- `Dataset/regresi_test.csv` (September–Oktober 2025, ~24 MB)
- `Dataset/klasifikasi_train.csv` & `klasifikasi_test.csv`
- `Dataset/clustering.csv`

*(Catatan: Direktori `Dataset/` diabaikan oleh Git via `.gitignore` karena batasan ukuran file GitHub >100 MB).*

---

## 📊 Ringkasan Teknis Task 1: Estimasi Waktu Kedatangan Paket (Regresi)

### 1. Temuan Kritis Audit Data & Eliminasi Data Leakage
- **Target Leakage Fatal pada `jumlah_scan`**: Berdasarkan `kamus_data.md`, `jumlah_scan` adalah akumulasi riwayat pelacakan selama perjalanan. Pada data latih (paket selesai), rata-ratanya adalah **9,65 scan**, sedangkan pada data uji (paket baru di-pickup), rata-ratanya hanya **1,50 scan**. Memasukkan fitur ini mengakibatkan lonjakan error Pinball Loss sebesar **+69,6%** saat diuji pada kondisi operasional riil. **Fitur ini wajib dieliminasi secara total.**
- **Missing Value & Nilai Sentinel**: Fitur `berat_kg` memiliki missing value 5,56% di train dan 5,44% di test. Karena `volume_liter` 100% lengkap dengan densitas fisik stabil (median 0,25 kg/L), nilai missing diimputasi menggunakan regresi volume. Nilai sentinel pada `jarak_km` (`0` dan `9999`) diganti dengan jarak median per rute.
- **Deteksi Unseen Category**: Kurir baru **`K13`** muncul sebanyak 2.342 kali (3,90%) pada data uji tetapi tidak pernah ada di data latih. Digunakan *smoothed target encoding* ($m=20$) agar otomatis *fallback* ke median global tanpa menimbulkan kesalahan runtime.

### 2. Desain Skema Validasi
- **Kegagalan Random K-Fold CV**: Validasi acak 5-Fold menghasilkan Pinball Loss **14,10 jam** (optimisme semu $-13,6\%$) akibat autokorelasi temporal.
- **Skema Temporal Hold-Out**: Memisahkan Train (Jan–Jun 2025, 178.694 paket) dan Validation (Jul–Agu 2025, 61.306 paket) menghasilkan evaluasi realistis **16,32 jam** yang mencerminkan generalisasi data uji (Sep–Okt 2025).

### 3. Pemodelan Regresi Kuantil & Kalibrasi Conformal
Evaluasi dilakukan pada metrik resmi **Pinball Loss** untuk kuantil median ($P_{50}$) dan batas atas ketidakpastian 90% ($P_{90}$):

| Model | Pinball Loss P50 (jam) | Pinball Loss P90 (jam) | Cakupan Empiris P90 (%) |
|---|:---:|:---:|:---:|
| Baseline (Median Historis per Layanan) | 14,32 | 13,32 | 89,95% |
| Ridge Regression (Log-Target) | 11,81 | 11,69 | 88,40% |
| **LightGBM Quantile (Raw)** | **10,71** | 11,32 | 87,10% |
| **LightGBM + Split Conformal Prediction** | **10,71** | **11,26** | **90,00% (Target Terpenuhi)** |

- **Kalibrasi Conformal**: Penyesuaian non-parametrik sebesar $\hat{q} = +4,467\text{ jam}$ menjamin tepat **90,00%** paket tiba sebelum waktu janji $P_{90}$, dengan kepatuhan syarat wajib $P_{90} \ge P_{50}$ pada 100% baris prediksi.

---

## ⚡ Reproduksi Angka Laporan (One-Command Reproduction)

Sesuai kriteria penilaian ujian lisan (reproduksi hasil dalam 5 menit), jalankan perintah berikut dari direktori utama:

```bash
python kode/task1_regresi.py
```

Skrip ini akan mengeksekusi pipeline lengkap secara otomatis, menghasilkan grafik evaluasi di `Task 1/figures/`, dan mencetak seluruh metrik kinerja serta memvalidasi kepatuhan file `prediksi_regresi.csv`.

---

## 👥 Tim Penyusun (Kelompok 8)
Lihat [kontribusi.md](kontribusi.md) untuk rincian pembagian kerja, kontribusi pull request, dan tanggung jawab masing-masing anggota.
